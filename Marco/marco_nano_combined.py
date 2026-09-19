
# Pin transformers <5.0 so Qwen3-MoE experts stay nn.Linear (bnb-quantizable).
# v5 fuses them into a 3-D nn.Parameter that bnb skips -> ~16 GB -> T4 OOM.

# Set BEFORE torch initialises CUDA. Hundreds of generations fragment the allocator;
# expandable_segments keeps the tight T4 headroom from OOMing late in the run.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
print("workdir:", BASE_DIR)

# OPTIONAL — only if the CSV isn't on Drive yet. Skip if you placed it manually.
import shutil

# Verify the CSV parses BEFORE loading several GB of weights.
import json, ast
from datasets import load_dataset


def _try_parse(s):
    """Rows may be JSON (double quotes) OR Python-repr (single quotes: {'role': ...})."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)   # handles the single-quoted repr format
        except Exception:
            return None


_ds = load_dataset("csv", data_files=str(BASE_DIR / "curated_dataset.csv"))["train"]
print("columns:", _ds.column_names)
print("rows   :", len(_ds))

# Pick the column whose first value parses into a list of role/content dicts.
DETECTED_COLUMN = None
for _c in _ds.column_names:
    _v = _try_parse(_ds[0][_c])
    if isinstance(_v, list) and _v and isinstance(_v[0], dict) and "role" in _v[0]:
        DETECTED_COLUMN = _c
        break

assert DETECTED_COLUMN, f"No conversation column found in {_ds.column_names}"
print("using column:", repr(DETECTED_COLUMN))

_first = _try_parse(_ds[0][DETECTED_COLUMN])
print(f"\nrow 0 parses to {len(_first)} message(s):")
for _m in _first:
    print("  ", _m.get("role"), "|", str(_m.get("content"))[:70])

_counts = {}
for _r in _ds:
    _p = _try_parse(_r[DETECTED_COLUMN])
    _n = len([m for m in _p if isinstance(m, dict) and m.get("role") == "user"]) if isinstance(_p, list) else -1
    _counts[_n] = _counts.get(_n, 0) + 1
print("\nuser-turns-per-row histogram:", _counts, " (-1 = failed to parse)")
print("(all rows showing 1 = single-turn data, as expected for the 'prompt' CSV)")

import torch

MODEL_ID = "AIDC-AI/Marco-Nano-Instruct"   # mirror: "ATH-MaaS/Marco-Nano-Instruct"

BASE_DIR       = BASE_DIR
LOCAL_DATASET  = BASE_DIR / "curated_dataset.csv"
CONV_COLUMN    = DETECTED_COLUMN

DEFAULT_SYSTEM_PROMPT = None   # no system prompt (matches ZAYA)

MEAN_LOG_DIR = BASE_DIR / "mean_pool_tensors" / "multi-turn" / "moe_routing_tensors"
TOPK_LOG_DIR = BASE_DIR / "topk_pool_tensors" / "multi-turn" / "moe_routing_tensors"
OUTPUT_RESULTS_FILE = BASE_DIR / "benchmark_generation_outputs.json"
LIVE_LOG_FILE       = BASE_DIR / "live_turns.jsonl"

MAX_TURNS      = 5      # ceiling only; shorter rows just end early
MAX_NEW_TOKENS = 2048    # Marco-Nano-Instruct is a non-reasoning instruct model
KMAX = 30   # keep top-K token softmax-probs per (layer,expert); K<=KMAX at build time

def _pool_topk(token_probs, KMAX=KMAX):
    T, E = token_probs.shape
    k = min(KMAX, T)
    vals = token_probs.topk(k, dim=0).values
    if k < KMAX:
        vals = torch.cat([vals, token_probs.new_zeros(KMAX - k, E)], dim=0)
    return vals
# ---- resumable extraction (CLI, like the ZAYA / LFM2 extractors) ------------
import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument("--start", type=int, default=1, help="1-based resume point")
_ap.add_argument("--limit", type=int, default=None, help="cap total conversations")
_ap.add_argument("--ids", type=str, default=None,
                 help="comma-separated 0-indexed conversation_ids to (re)extract")
_args, _ = _ap.parse_known_args()

TEST_LIMIT     = 2000
if _args.limit is not None:
    TEST_LIMIT = _args.limit
START_INDEX = max(1, _args.start)
SELECTED_IDS = None
if _args.ids:
    SELECTED_IDS = set(int(x) for x in _args.ids.replace(" ", "").split(",") if x != "")
    print(f"[ids] targeting {len(SELECTED_IDS)} conversations")
SEED           = 1234

MEAN_LOG_DIR.mkdir(parents=True, exist_ok=True)
TOPK_LOG_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
print("outputs ->", BASE_DIR)

import gc
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained(MODEL_ID)

NUM_EXPERTS = (getattr(cfg, "num_experts", None)
               or getattr(cfg, "num_local_experts", None)
               or getattr(cfg, "n_routed_experts", None))
TOP_K   = getattr(cfg, "num_experts_per_tok", 8)
MAX_CTX = getattr(cfg, "max_position_embeddings", 8192)

print(f"model_type={cfg.model_type}  num_experts={NUM_EXPERTS}  "
      f"top_k={TOP_K}  max_ctx={MAX_CTX}  layers={cfg.num_hidden_layers}")

with init_empty_weights():
    _probe = AutoModelForCausalLM.from_config(cfg)

ROUTER_NAMES = [n for n, m in _probe.named_modules()
                if getattr(m, "out_features", None) == NUM_EXPERTS and "experts" not in n]

_expert_mods = [n for n, m in _probe.named_modules() if n.endswith("experts")]

del _probe
gc.collect()

print(f"\nfound {len(ROUTER_NAMES)} routers")
for n in ROUTER_NAMES[:3]:
    print("   ", n)
print("expert container example:", _expert_mods[0] if _expert_mods else "(none)")

assert ROUTER_NAMES, (
    "No router modules found. Inspect module names manually and set ROUTER_NAMES by hand."
)

from transformers import AutoTokenizer, BitsAndBytesConfig

# Exact router names -> no substring collision with expert gate_proj. Routers stay fp16 so
# the logits you are studying are undistorted; costs a few MB. lm_head is tied to the
# embeddings here, so leaving it unquantized is free.
SKIP_MODULES = ROUTER_NAMES + ["lm_head"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=SKIP_MODULES,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)   # no trust_remote_code

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager",
    )
except TypeError:
    # older transformers spells it torch_dtype=
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager",
    )

model.eval()
fp = model.get_memory_footprint() / 1e9
print("footprint:", round(fp, 3), "GB")
if fp > 9:
    print("!! >9 GB means the experts did NOT quantize (likely transformers>=5). "
          "Pin transformers<5.0 and restart, or this will OOM the T4.")

# Terminators. Qwen3/ChatML stops on <|im_end|>; without it every generation runs to the
# token cap and trails garbage.
if getattr(model.generation_config, "top_k", None) is not None and model.generation_config.top_k <= 0:
    model.generation_config.top_k = None
if getattr(model.generation_config, "top_p", None) is not None and not (0.0 < model.generation_config.top_p <= 1.0):
    model.generation_config.top_p = None
model.generation_config.do_sample = True

terminators = set()
if tokenizer.eos_token_id is not None:
    terminators.add(tokenizer.eos_token_id)
for tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>"]:
    tid = tokenizer.convert_tokens_to_ids(tok)
    if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
        terminators.add(tid)
terminators = sorted(terminators)

pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

print("terminators:", [(t, tokenizer.convert_ids_to_tokens(t)) for t in terminators])
print("pad_id:", pad_id, " max_ctx:", MAX_CTX)

# Precision census - confirms the experts actually quantized (the whole ballgame on T4).
from collections import defaultdict

d = defaultdict(int)
for n, p in model.named_parameters():
    d[(type(p).__name__, str(p.dtype))] += p.numel()
for k, v in sorted(d.items(), key=lambda x: -x[1]):
    print(f"{v/1e9:7.3f}B  {k}")

_r = model.get_submodule(ROUTER_NAMES[0])
print(f"\nrouter[0] {ROUTER_NAMES[0]}")
print("   type :", type(_r).__name__)
print("   dtype:", next(_r.parameters()).dtype, " <- want float16, NOT uint8")

import re
import torch.nn.functional as F

# Qwen3-MoE routing: gate projects hidden_size -> NUM_EXPERTS, softmax over experts, then
# top-k with norm_topk_prob renormalising the chosen k. The gate module emits the full
# per-expert logits directly, so the hook usually reads them straight off the output.

logit_capture    = {}   # layer_n -> [ (tokens, E) chunks ]
layer_num_experts = {}
_shape_diag = {"printed": False}


def _to_2d(t):
    if t.dim() == 3:
        return t.reshape(-1, t.shape[-1])
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() == 2:
        return t
    return None


def _logits_from_output(output, E):
    """Return a full-logit tensor (last dim == E) if the module output already has one."""
    cands = output if isinstance(output, (tuple, list)) else (output,)
    for c in cands:
        if torch.is_tensor(c) and c.shape[-1] == E and c.is_floating_point():
            return c
    return None


def _find_gate_weight(module, E):
    """Router projection weight (E, hidden), full precision.

    Returns None for a bnb-quantized Linear (packed uint8); since routers are in
    SKIP_MODULES this should resolve to the fp16 weight.
    """
    for _, p in module.named_parameters(recurse=True):
        if p.dim() == 2 and p.shape[0] == E and p.is_floating_point():
            return p
    return None


def _make_router_hook(layer_n, gate_weight):
    def hook(module, inputs, output):
        # Preferred: module already emits full per-expert logits (plain nn.Linear gate).
        t = _logits_from_output(output, NUM_EXPERTS)
        # Fallback: recompute full logits from the router INPUT and its gate weight.
        if t is None:
            x = inputs[0] if inputs else None
            if x is None or gate_weight is None:
                return
            t = F.linear(x.to(gate_weight.dtype), gate_weight)
        if not torch.is_tensor(t):
            return
        if not _shape_diag["printed"]:
            print(f"[diag] router L{layer_n} logits shape={tuple(t.shape)} dtype={t.dtype}")
            _shape_diag["printed"] = True
        t = _to_2d(t.detach().float().cpu())
        if t is None:
            return
        layer_num_experts[layer_n] = t.shape[-1]
        logit_capture.setdefault(layer_n, []).append(t)
    return hook


for h in globals().get("_hook_handles", []):
    h.remove()
_hook_handles = []
router_layers = []

for name in ROUTER_NAMES:
    module = model.get_submodule(name)
    m = re.search(r"layers\.(\d+)\.", name)
    layer_n = int(m.group(1)) if m else len(router_layers)
    gate_weight = _find_gate_weight(module, NUM_EXPERTS)
    _hook_handles.append(module.register_forward_hook(_make_router_hook(layer_n, gate_weight)))
    router_layers.append(layer_n)

print(f"Registered {len(_hook_handles)} router hooks on layers: {router_layers}")
print("gate_weight resolved for layer 0:",
      _find_gate_weight(model.get_submodule(ROUTER_NAMES[0]), NUM_EXPERTS) is not None)

def extract_and_pool_routing(prompt_len, conversation_id, turn_id):
    """Isolate generated-response rows, mean-pool over tokens, save to disk.

    Prefill contributes `prompt_len` rows; each decode step contributes 1. Rows
    [prompt_len:] are the routing decisions taken while producing the new tokens.

    Qwen3-MoE gates with SOFTMAX over experts and activates TOP_K per token, so:
      * pooled_probs   = mean over tokens of softmax(logits)
      * topk_histogram = how often each expert lands in a token's top-K.
        Softmax is monotonic, so top-k on raw logits picks the same experts.
    """
    layers = sorted(logit_capture.keys())
    if not layers:
        return None, {"tensor_matrix_path": None}

    pooled_logits, pooled_probs, topk, hist = {}, {}, {}, {}
    resp_tokens = 0

    for layer_n in layers:
        logits = torch.cat(logit_capture[layer_n], dim=0)   # (num_tokens, E_layer)
        E = logits.shape[-1]
        if logits.shape[0] <= prompt_len:
            pooled_logits[layer_n] = torch.zeros(E)
            pooled_probs[layer_n]  = torch.zeros(E)
            topk[layer_n] = torch.zeros(KMAX, E)
            hist[layer_n] = [0] * E
            continue
        resp = logits[prompt_len:, :]
        resp_tokens = resp.shape[0]
        pooled_logits[layer_n] = resp.mean(dim=0)
        resp_sm = torch.softmax(resp, dim=-1)
        pooled_probs[layer_n]  = resp_sm.mean(dim=0)
        topk[layer_n] = _pool_topk(resp_sm, KMAX)
        k = min(TOP_K, E)
        topk_idx = resp.topk(k, dim=-1).indices.reshape(-1)
        hist[layer_n] = torch.bincount(topk_idx, minlength=E).tolist()

    e_max = max(v.shape[0] for v in pooled_logits.values())

    def _stack(dct):
        rows = []
        for layer_n in layers:
            v = dct[layer_n]
            pad = e_max - v.shape[0]
            rows.append(F.pad(v, (0, pad)) if pad else v)
        return torch.stack(rows)

    matrix      = _stack(pooled_logits)     # (num_moe_layers, E_max)
    prob_matrix = _stack(pooled_probs)
    mask = [[1] * layer_num_experts[l] + [0] * (e_max - layer_num_experts[l]) for l in layers]

    def _stack_topk(d):
        rows = []
        for layer_n in layers:
            v = d[layer_n]
            pad = e_max - v.shape[-1]
            rows.append(F.pad(v, (0, pad)) if pad else v)
        return torch.stack(rows)            # (L, KMAX, E_max)
    topk_matrix = _stack_topk(topk)

    base_payload = {
        "pooled_logits": matrix,
        "pooled_probs": prob_matrix,
        "layer_indices": layers,
        "num_experts_per_layer": [layer_num_experts[l] for l in layers],
        "valid_mask": torch.tensor(mask, dtype=torch.bool),
        "response_token_count": resp_tokens,
        "top_k": TOP_K,
        "topk_expert_histogram": hist,
        "gating": "softmax",
    }

    fname = f"conv_{conversation_id:04d}_turn_{turn_id:02d}.pt"
    torch.save(base_payload, MEAN_LOG_DIR / fname)
    topk_payload = dict(base_payload)
    topk_payload["topk_token_probs"] = topk_matrix
    torch.save(topk_payload, TOPK_LOG_DIR / fname)

    meta = {
        "tensor_matrix_path": fname,
        "routing_matrix_shape": list(matrix.shape),
        "routed_response_tokens": resp_tokens,
        "moe_layer_indices": layers,
    }
    return fname, meta

import ast


def parse_conversation(raw):
    """Row value -> list of {"role","content"} dicts.

    Handles an already-decoded list, a JSON string, a Python-repr string,
    ShareGPT-style {"from","value"}, and odd capitalisation.
    """
    if raw is None:
        return []
    obj = raw
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                return []
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        return []

    msgs = []
    for m in obj:
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                continue
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", m.get("from", ""))).strip().lower()
        content = m.get("content", m.get("value", m.get("text", "")))
        if not isinstance(content, str):
            content = str(content)
        if role in ("human", "usr"):
            role = "user"
        elif role in ("gpt", "bot", "ai", "model"):
            role = "assistant"
        content = content.strip()
        if role and content:
            msgs.append({"role": role, "content": content})
    return msgs


def _apply_template(history, **extra):
    return tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True, **extra)


def build_prompt(history):
    """Render with the model's own chat template (Qwen3 ChatML).

    enable_thinking=False keeps Marco-Nano out of any <think> preamble so prompt_len stays
    aligned with the router rows. Tokenised later with add_special_tokens=False: the template
    already carries role markers, and an auto-BOS would shift the prefill/response split.
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            try:
                return _apply_template(history, enable_thinking=False)
            except TypeError:
                return _apply_template(history)
        except Exception:
            # Some templates reject a bare system role - fold it into the first user turn.
            if history and history[0]["role"] == "system":
                sys_txt = history[0]["content"]
                folded = [dict(m) for m in history[1:]]
                if folded and folded[0]["role"] == "user":
                    folded[0]["content"] = f"{sys_txt}\n\n{folded[0]['content']}"
                try:
                    return _apply_template(folded, enable_thinking=False)
                except TypeError:
                    return _apply_template(folded)
            raise
    # Manual ChatML fallback.
    text = ""
    for msg in history:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


# Preview the exact string the model will see.
_demo = [
         {"role": "user", "content": "Prompt 1"},
         {"role": "assistant", "content": "Reply 1"},
         {"role": "user", "content": "Prompt 2"}]
print(repr(build_prompt(_demo)))

def flush_results(results_log):
    """Atomic rewrite - the file is never half-written if you open it mid-run."""
    tmp = OUTPUT_RESULTS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUTPUT_RESULTS_FILE)


_TRUNCATED_TURNS = []

def remove_conv_from_live(conv_id):
    if not LIVE_LOG_FILE.exists():
        return
    kept = []
    with open(LIVE_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line); continue
            if rec.get("conversation_id") != conv_id:
                kept.append(line)
    with open(LIVE_LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(kept)

def append_live(record):
    with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def run_dataset_benchmark():
    print("Loading dataset...")
    path = Path(LOCAL_DATASET).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    builder = {".csv": "csv", ".tsv": "csv", ".json": "json",
               ".jsonl": "json", ".parquet": "parquet"}.get(path.suffix.lower())
    if builder is None:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")
    loaded = load_dataset(builder, data_files=path.as_posix())
    split_name = list(loaded.keys())[0]
    dataset = loaded[split_name]
    print(f"Using split '{split_name}' ({len(dataset)} rows)")

    if CONV_COLUMN not in dataset.column_names:
        raise KeyError(f"Column {CONV_COLUMN!r} not found. Available: {dataset.column_names}")

    # Truncate the live log only on a fresh run; a resume appends.
    if START_INDEX <= 1 and SELECTED_IDS is None:
        open(LIVE_LOG_FILE, "w").close()

    results_log = []
    for conv_idx, sample in enumerate(dataset):
        if SELECTED_IDS is not None:
            if conv_idx not in SELECTED_IDS:
                continue
        else:
            if conv_idx >= TEST_LIMIT:
                break
            if (conv_idx + 1) < START_INDEX:
                continue
            _existing = MEAN_LOG_DIR / f"conv_{conv_idx:04d}_turn_01.pt"
            if _existing.exists():
                print(f"  [skip] conversation {conv_idx + 1}: exists")
                continue

        messages = parse_conversation(sample.get(CONV_COLUMN))
        user_turns = [m["content"] for m in messages if m["role"] == "user"]

        print(f"\n{'=' * 70}\nCONVERSATION {conv_idx + 1}/{TEST_LIMIT}")
        print(f"parsed {len(messages)} messages -> {len(user_turns)} user turns")

        if not user_turns:
            print("  Warning: no user prompts in this row. Skipping.")
            continue

        record = {
            "conversation_id": conv_idx,
            "system_prompt": None,
            "dataset_user_turns": len(user_turns),
            "status": "running",
            "turns": [],
        }
        results_log.append(record)
        flush_results(results_log)

        history = []   # no system prompt (matches ZAYA)

        for turn_idx, user_prompt in enumerate(user_turns):
            if turn_idx >= MAX_TURNS:
                record["status"] = f"halted_max_turns({MAX_TURNS})"
                print(f"  Reached MAX_TURNS ({MAX_TURNS}); {len(user_turns) - turn_idx} prompts unused.")
                break

            history.append({"role": "user", "content": user_prompt})
            prompt_text = build_prompt(history)
            inputs = tokenizer(prompt_text, return_tensors="pt",
                               add_special_tokens=False).to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            if prompt_len + MAX_NEW_TOKENS >= MAX_CTX:
                history.pop()
                record["status"] = f"halted_context_limit(turn={turn_idx + 1}, len={prompt_len})"
                print(f"  Context ceiling hit at turn {turn_idx + 1} ({prompt_len} tok). Halting.")
                break

            print(f"\n--- TURN {turn_idx + 1}/{len(user_turns)} | prompt_len={prompt_len} ---")
            print(f"USER : {user_prompt[:300]}")

            logit_capture.clear()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    eos_token_id=terminators,
                    pad_token_id=pad_id,
                )

            new_ids = outputs[0][prompt_len:]
            response_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            gen_len = int(new_ids.shape[0])
            # non-reasoning model: truncation = hit the token cap
            is_truncated = gen_len >= MAX_NEW_TOKENS - 1

            print(f"MODEL: {response_text[:300]}")

            if is_truncated:
                logit_capture.clear()
                _TRUNCATED_TURNS.append((conv_idx, turn_idx + 1, gen_len))
                print(f"  !! TRUNCATED at {gen_len}/{MAX_NEW_TOKENS} "
                      f"-- SKIPPING conversation {conv_idx} entirely.")
                for _pt in range(1, turn_idx + 1):
                    for _d in (MEAN_LOG_DIR, TOPK_LOG_DIR):
                        _f = _d / f"conv_{conv_idx:04d}_turn_{_pt:02d}.pt"
                        if _f.exists():
                            _f.unlink()
                if record in results_log:
                    results_log.remove(record)
                    flush_results(results_log)
                remove_conv_from_live(conv_idx)
                record = None
                del outputs, inputs
                torch.cuda.empty_cache()
                break

            tensor_file, routing_meta = extract_and_pool_routing(prompt_len, conv_idx, turn_idx + 1)
            logit_capture.clear()

            if not response_text:
                print("  !! EMPTY GENERATION - check the chat template / terminators.")

            if response_text:
                history.append({"role": "assistant", "content": response_text})

            turn_record = {
                "turn_id": turn_idx + 1,
                "user_prompt": user_prompt,
                "model_response": response_text,
                "prompt_token_len": prompt_len,
                "generated_token_len": gen_len,
                "truncated": is_truncated,
                "raw_prompt_fed_to_model": prompt_text,
                **routing_meta,
            }
            record["turns"].append(turn_record)
            record["running_history"] = list(history)
            flush_results(results_log)
            append_live({"conversation_id": conv_idx, **turn_record})
            print(f"  saved -> {tensor_file}")

            del outputs, inputs
            torch.cuda.empty_cache()

        if record is not None:
            if record["status"] == "running":
                record["status"] = "complete"
            record["final_history"] = history
            flush_results(results_log)

    print(f"\nComplete.\n  Full results : {OUTPUT_RESULTS_FILE}\n  Live stream  : {LIVE_LOG_FILE}")


if __name__ == "__main__":
    run_dataset_benchmark()
