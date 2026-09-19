import os
import re
import json
import ast
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

# ==========================================
# PHASE 1: CONFIGURATION & MODEL LOADING
# ==========================================
# LFM2.5-8B-A1B is natively supported in recent transformers as the `lfm2moe`
# architecture (Lfm2MoeForCausalLM). You need a transformers version new enough
# to include it; otherwise from_pretrained raises "model type `lfm2moe` not
# recognized" and you must `pip install -U transformers`.
MODEL_ID = "LiquidAI/LFM2.5-8B-A1B"

# Everything is anchored to the SCRIPT's folder, not the current working
# directory, so `python lfm2.py` behaves the same no matter where you launch it.
BASE_DIR = Path(__file__).resolve().parent

# --- dataset: use ONE of these two ---
# (a) local file  -> point LOCAL_DATASET at it, leave DATASET_ID as None
# (b) hub repo id -> set DATASET_ID, leave LOCAL_DATASET as None
LOCAL_DATASET = BASE_DIR / "curated_dataset.csv"
DATASET_ID = None
DATASET_SPLIT = None       # None -> take whatever split exists

CONV_COLUMN = "prompt"   # the only column used

# NOTE: no system prompt. The conversation starts with the first user turn, and any
# system message that happens to appear in a row is ignored (see PHASE 5).

MEAN_LOG_DIR = BASE_DIR / "mean_pool_tensors" / "multi-turn" / "moe_routing_tensors"
TOPK_LOG_DIR = BASE_DIR / "topk_pool_tensors" / "multi-turn" / "moe_routing_tensors"
OUTPUT_RESULTS_FILE = BASE_DIR / "benchmark_generation_outputs.json"  # rewritten EVERY turn
LIVE_LOG_FILE = BASE_DIR / "live_turns.jsonl"                         # appended EVERY turn

MAX_TURNS = 5              # ceiling only; a row with fewer prompts just ends early
KMAX = 30   # keep the top-K token gate-scores per (layer, expert); any K<=KMAX
            # is a feature-build knob later. Adds payload key: topk_token_probs.

def _pool_topk(token_scores, KMAX=KMAX):
    """(T, E) sigmoid gate scores -> (KMAX, E): the KMAX highest token scores
    per expert, sorted descending, zero-padded when T < KMAX. Mirrors the ZAYA
    combined extractor so any K<=KMAX is selectable at feature-build time."""
    import torch as _t
    T, E = token_scores.shape
    k = min(KMAX, T)
    vals = token_scores.topk(k, dim=0).values          # (k, E) desc over tokens
    if k < KMAX:
        vals = _t.cat([vals, token_scores.new_zeros(KMAX - k, E)], dim=0)
    return vals

MAX_NEW_TOKENS = 1024       # NOTE: LFM2.5 is a reasoning model (emits chain-of-thought
                           # before the final answer), so you may want to raise this.
TEST_LIMIT = 2000

# ---- resumable extraction ---------------------------------------------------
# python lfm2_multi_turn_combined.py --start 16 --limit 2000
#   --start N   : 1-based; skip conversations before N (resume after a crash).
#   --limit N   : cap total conversations.
#   --ids a,b,c : extract ONLY these conversation_ids (0-indexed), overwriting.
# A conversation whose mean tensor already exists is skipped, so a re-run
# continues instead of restarting from 0.
import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument("--start", type=int, default=1)
_ap.add_argument("--limit", type=int, default=None)
_ap.add_argument("--ids", type=str, default=None,
                 help="comma-separated 0-indexed conversation_ids to (re)extract")
_args, _ = _ap.parse_known_args()
START_INDEX = max(1, _args.start)
if _args.limit is not None:
    TEST_LIMIT = _args.limit
SELECTED_IDS = None
if _args.ids:
    SELECTED_IDS = set(int(x) for x in _args.ids.replace(" ", "").split(",") if x != "")
    print(f"[ids] targeting {len(SELECTED_IDS)} conversations: "
          f"{sorted(SELECTED_IDS)[:10]}{'...' if len(SELECTED_IDS) > 10 else ''}")
SEED = 1234

MEAN_LOG_DIR.mkdir(parents=True, exist_ok=True)
TOPK_LOG_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)

print(f"Loading {MODEL_ID} strictly to GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    ),
    # For a routing STUDY on a 16 GB card, 8-bit is a better fidelity/VRAM trade
    # (8.3B in 8-bit ~= 9 GB, well within your headroom). To switch, comment out
    # the block above and use:  quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    trust_remote_code=True,
)
model.eval()

if getattr(model.generation_config, "top_k", None) is not None and model.generation_config.top_k <= 0:
    model.generation_config.top_k = None
if getattr(model.generation_config, "top_p", None) is not None and not (0.0 < model.generation_config.top_p <= 1.0):
    model.generation_config.top_p = None

terminators = set()
if tokenizer.eos_token_id is not None:
    terminators.add(tokenizer.eos_token_id)
for tok in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>"]:
    tid = tokenizer.convert_tokens_to_ids(tok)
    if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
        terminators.add(tid)
terminators = sorted(terminators)

model.generation_config.do_sample = True   # silences the "top_p/top_k not valid" warning

pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
MAX_CTX = getattr(model.config, "max_position_embeddings", 8192)

# ==========================================
# PHASE 2: ROUTER HOOKS  (LFM2-MoE-SPECIFIC)
# ==========================================
# Real structure (confirmed from the run + the transformers source):
#   model.layers.{n}.feed_forward   -> Lfm2MoeSparseMoeBlock  (MoE layers only; the
#                                      first `num_dense_layers` layers are dense
#                                      Lfm2MoeMLP -- which is why hooks start at layer 2)
#       .gate     -> the router. A plain nn.Linear in some transformers versions,
#                    or an Lfm2MoeTopKRouter WRAPPER in newer ones.
#       .experts  -> Lfm2MoeExperts (fused/batched experts)
#
# The router projects hidden_size -> num_experts (=32) to get per-token logits, then
# routes with SIGMOID scores + top-k (k = num_experts_per_tok = 4). Two consequences:
#
#   1) The wrapper (Lfm2MoeTopKRouter) forward returns ONLY the top-k (indices,
#      weights) -- NOT the full 32-wide logits your pooling needs. So when the module
#      output doesn't contain a 32-wide tensor, we RECOMPUTE the full logits from the
#      router's INPUT and its gate weight:  logits = input @ W_gate.T . This works
#      whether the gate is an nn.Linear (we just use its output) or a wrapper.
#
#   2) LFM2 gates with sigmoid, not softmax -- so pooled "probs" below are sigmoid,
#      and the histogram counts the top-k selected experts (not a single argmax).

NUM_EXPERTS = (
    getattr(model.config, "num_experts", None)
    or getattr(model.config, "num_local_experts", None)
    or getattr(model.config, "n_routed_experts", None)
)
TOP_K = getattr(model.config, "num_experts_per_tok", 4)
print(f"[diag] num_experts={NUM_EXPERTS}  top_k={TOP_K}  num_dense_layers="
      f"{getattr(model.config, 'num_dense_layers', '?')}")

logit_capture = {}        # layer_n -> [ (tokens, E) chunks ]
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
    """The router's projection weight, shape (E, hidden), full precision.

    Returns None for a bnb-quantized Linear (weight is packed uint8, not floating),
    in which case the hook falls back to the module's own float output.
    """
    for _, p in module.named_parameters(recurse=True):
        if p.dim() == 2 and p.shape[0] == E and p.is_floating_point():
            return p
    return None


def _make_router_hook(layer_n, gate_weight):
    def hook(module, inputs, output):
        # Preferred: the module already emits full per-expert logits (nn.Linear gate).
        t = _logits_from_output(output, NUM_EXPERTS)
        # Fallback: wrapper router (Lfm2MoeTopKRouter) that returns only top-k ->
        # recompute the full logits from the router INPUT and its gate weight.
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


router_layers = []
for name, module in model.named_modules():
    # The router lives at ...feed_forward.gate in every layer that is MoE.
    if not name.endswith("feed_forward.gate"):
        continue
    m = re.search(r"layers\.(\d+)\.", name)
    layer_n = int(m.group(1)) if m else len(router_layers)
    # Resolve a full-precision gate weight for the wrapper/quantized path (may be None
    # for a plain fp nn.Linear, where we use the module's float output instead).
    gate_weight = _find_gate_weight(module, NUM_EXPERTS)
    module.register_forward_hook(_make_router_hook(layer_n, gate_weight))
    router_layers.append(layer_n)

print(f"Registered {len(router_layers)} LFM2-MoE router hooks on layers: {router_layers}")
if not router_layers:
    print("!! No router hooks attached. Candidate module names:")
    for name, module in model.named_modules():
        if any(k in name for k in ("gate", "router", "moe", "expert")):
            print("   ", name, "->", type(module).__name__,
                  f"(out_features={getattr(module, 'out_features', '?')})")
    raise RuntimeError("Router modules not found — adjust the name match in PHASE 2.")


def extract_and_pool_routing(prompt_len, conversation_id, turn_id):
    """
    Concatenate per-step router logits, isolate the generated-response rows,
    mean-pool over the token axis, and save.

    Prefill contributes `prompt_len` rows; each decode step contributes 1 row.
    Rows [prompt_len:] are therefore the routing decisions taken while processing
    the newly generated tokens.

    LFM2-MoE gates with SIGMOID and activates TOP_K experts/token, so:
      * pooled_probs   = mean over tokens of sigmoid(logits)   (per-expert gate score;
                         these are independent gates and do NOT sum to 1)
      * topk_histogram = count of how often each expert lands in the token's top-K.
        (This uses the sigmoid-score top-K and ignores the optional `expert_bias`
        additive term used during selection, so it is a faithful approximation of,
        not a byte-exact replica of, the model's routed set.)
    """
    layers = sorted(logit_capture.keys())
    if not layers:
        return None, {"tensor_matrix_path": None}

    pooled_logits = {}   # mean-pooled raw logits
    pooled_probs = {}    # mean-pooled sigmoid gate scores
    topk = {}            # (KMAX, E) top-K token gate-scores per expert  <-- NEW
    hist = {}            # top-K expert selection counts over the response tokens
    resp_tokens = 0

    for layer_n in layers:
        logits = torch.cat(logit_capture[layer_n], dim=0)     # (num_tokens, E_layer)
        E = logits.shape[-1]

        if logits.shape[0] <= prompt_len:
            pooled_logits[layer_n] = torch.zeros(E)
            pooled_probs[layer_n] = torch.zeros(E)
            topk[layer_n] = torch.zeros(KMAX, E)
            hist[layer_n] = [0] * E
            continue

        resp = logits[prompt_len:, :]
        resp_tokens = resp.shape[0]

        pooled_logits[layer_n] = resp.mean(dim=0)
        resp_sig = torch.sigmoid(resp)                                   # (T, E) sigmoid gates
        pooled_probs[layer_n] = resp_sig.mean(dim=0)                     # mean pooling
        topk[layer_n] = _pool_topk(resp_sig, KMAX)                       # (KMAX, E) top-K pooling
        k = min(TOP_K, E)
        topk_idx = resp.topk(k, dim=-1).indices.reshape(-1)             # top-K per token
        hist[layer_n] = torch.bincount(topk_idx, minlength=E).tolist()

    # Expert counts are constant (32) across LFM2-MoE layers, so this pad is a no-op
    # here -- kept intact so the routine also handles ragged architectures.
    e_max = max(v.shape[0] for v in pooled_logits.values())

    def _stack(d):
        rows = []
        for layer_n in layers:
            v = d[layer_n]
            pad = e_max - v.shape[0]
            rows.append(torch.nn.functional.pad(v, (0, pad)) if pad else v)
        return torch.stack(rows)

    matrix = _stack(pooled_logits)          # (num_moe_layers, E_max)
    prob_matrix = _stack(pooled_probs)
    mask = [[1] * layer_num_experts[l] + [0] * (e_max - layer_num_experts[l]) for l in layers]

    # stack top-K token scores to (num_moe_layers, KMAX, E_max), padding experts axis
    def _stack_topk(d):
        rows = []
        for layer_n in layers:
            v = d[layer_n]                      # (KMAX, E_layer)
            pad = e_max - v.shape[-1]
            rows.append(torch.nn.functional.pad(v, (0, pad)) if pad else v)
        return torch.stack(rows)                # (L, KMAX, E_max)
    topk_matrix = _stack_topk(topk)

    # shared base payload (identical for both poolings — one generation)
    base_payload = {
        "pooled_logits": matrix,
        "pooled_probs": prob_matrix,           # sigmoid gate scores (mean pooling)
        "layer_indices": layers,
        "num_experts_per_layer": [layer_num_experts[l] for l in layers],
        "valid_mask": torch.tensor(mask, dtype=torch.bool),
        "response_token_count": resp_tokens,
        "top_k": TOP_K,
        "topk_expert_histogram": hist,
    }

    fname = f"conv_{conversation_id:04d}_turn_{turn_id:02d}.pt"

    # MEAN payload -> mean_pool_tensors (build_dataset reads pooled_probs for mean)
    torch.save(base_payload, MEAN_LOG_DIR / fname)

    # TOPK payload -> topk_pool_tensors (base + the top-K token scores for K sweep)
    topk_payload = dict(base_payload)
    topk_payload["topk_token_probs"] = topk_matrix     # (L, KMAX, E)
    torch.save(topk_payload, TOPK_LOG_DIR / fname)

    meta = {
        "tensor_matrix_path": fname,
        "routing_matrix_shape": list(matrix.shape),
        "routed_response_tokens": resp_tokens,
        "moe_layer_indices": layers,
    }
    return fname, meta


# ==========================================
# PHASE 3: DATASET PARSING
# ==========================================
def parse_conversation(raw):
    """
    conversation cell -> list of {"role","content"} dicts.
    Handles: an already-decoded list, a JSON string, a Python-repr string,
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


def build_prompt(history):
    """
    Use the model's own chat template when available. Tokenise with
    add_special_tokens=False: the template already carries BOS/role markers, and a
    second auto-BOS would shift prompt_len out of sync with the router rows.

    `history` contains only user/assistant messages -- no system role -- so the
    template is applied directly.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

    text = tokenizer.bos_token or ""
    for msg in history:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


# ==========================================
# PHASE 4: INCREMENTAL WRITERS
# ==========================================
def flush_results(results_log):
    """Atomic rewrite, so the file is never half-written when you open it mid-run."""
    tmp = OUTPUT_RESULTS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUTPUT_RESULTS_FILE)


_TRUNCATED_TURNS = []   # (conv_idx, turn_id, gen_len) for conversations skipped

def append_live(record):
    with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())



def remove_conv_from_live(conv_id):
    """Strip all lines for a conversation_id from live_turns.jsonl (used when a
    conversation is skipped mid-way; earlier turns may already be written)."""
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

# ==========================================
# PHASE 5: EVALUATION LOOP
# ==========================================
def run_dataset_benchmark():
    print("Loading dataset...")
    if LOCAL_DATASET is not None:
        path = Path(LOCAL_DATASET).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        # .as_posix() -> forward slashes; `datasets` mishandles Windows backslashes
        builder = {".csv": "csv", ".tsv": "csv", ".json": "json",
                   ".jsonl": "json", ".parquet": "parquet"}.get(path.suffix.lower())
        if builder is None:
            raise ValueError(f"Unsupported dataset extension: {path.suffix}")
        print(f"Loading local dataset: {path}")
        loaded = load_dataset(builder, data_files=path.as_posix())
    else:
        loaded = load_dataset(DATASET_ID)

    if DATASET_SPLIT:
        dataset = loaded[DATASET_SPLIT]
    else:
        # No named split -> take the only one there is.
        split_name = list(loaded.keys())[0]
        dataset = loaded[split_name]
        print(f"Using split '{split_name}' ({len(dataset)} rows)")

    if CONV_COLUMN not in dataset.column_names:
        raise KeyError(
            f"Column {CONV_COLUMN!r} not found. Available: {dataset.column_names}"
        )

    if START_INDEX <= 1 and SELECTED_IDS is None:
        open(LIVE_LOG_FILE, "w").close()
    results_log = []

    for conv_idx, sample in enumerate(dataset):
        # --- selection / resume gating ---
        if SELECTED_IDS is not None:
            if conv_idx not in SELECTED_IDS:
                continue
            # targeted re-extraction overwrites existing tensors (the point)
        else:
            if conv_idx >= TEST_LIMIT:
                break
            if (conv_idx + 1) < START_INDEX:
                continue
            _existing = MEAN_LOG_DIR / f"conv_{conv_idx:04d}_turn_01.pt"
            if _existing.exists():
                print(f"  [skip] conversation {conv_idx + 1}: {_existing.name} exists")
                continue

        messages = parse_conversation(sample.get(CONV_COLUMN))

        # Replay ONLY the user prompts; the model writes its own assistant turns.
        # Any system message in the row is ignored -- no system prompt is used.
        user_turns = [m["content"] for m in messages if m["role"] == "user"]

        print(f"\n{'=' * 70}\nCONVERSATION {conv_idx + 1}/{TEST_LIMIT}")
        print(f"parsed {len(messages)} messages -> {len(user_turns)} user turns")

        if not user_turns:
            print("  Warning: no user prompts in this row. Skipping.")
            continue

        record = {
            "conversation_id": conv_idx,
            "dataset_user_turns": len(user_turns),
            "status": "running",
            "turns": [],
        }
        results_log.append(record)
        flush_results(results_log)          # visible before the first token is generated

        # No system message: the conversation starts with the first user turn.
        history = []

        # Replay each user prompt in order. When the row's prompts run out, the
        # conversation is done -- move on to the next row. MAX_TURNS is just a
        # ceiling for unusually long rows.
        for turn_idx, user_prompt in enumerate(user_turns):

            if turn_idx >= MAX_TURNS:
                record["status"] = f"halted_max_turns({MAX_TURNS})"
                print(f"  Reached MAX_TURNS ({MAX_TURNS}); {len(user_turns) - turn_idx} prompts unused.")
                break

            history.append({"role": "user", "content": user_prompt})

            prompt_text = build_prompt(history)
            inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
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
            # LFM2 closes </think> early then keeps writing the answer, so a capped
            # response usually HAS </think> yet is still cut off mid-answer. Detect
            # truncation purely by hitting the token cap (a finished response stops
            # before the cap via EOS).
            is_truncated = gen_len >= MAX_NEW_TOKENS - 1

            print(f"MODEL: {response_text[:300]}")

            # --- SKIP the whole conversation on truncation ---
            # A truncated turn makes the conversation unusable (stateful multi-turn).
            # Do NOT extract/save its tensors; drop it from tensors, JSON, and JSONL,
            # and move to the next conversation.
            if is_truncated:
                logit_capture.clear()
                _TRUNCATED_TURNS.append((conv_idx, turn_idx + 1, gen_len))
                print(f"  !! TRUNCATED at {gen_len}/{MAX_NEW_TOKENS} (no </think>) "
                      f"-- SKIPPING conversation {conv_idx} entirely.")
                # (a) remove tensors already written for earlier turns of this conv
                for _pt in range(1, turn_idx + 1):
                    for _d in (MEAN_LOG_DIR, TOPK_LOG_DIR):
                        _f = _d / f"conv_{conv_idx:04d}_turn_{_pt:02d}.pt"
                        if _f.exists():
                            _f.unlink()
                # (b) remove this conversation from the benchmark JSON
                if record in results_log:
                    results_log.remove(record)
                    flush_results(results_log)
                # (c) remove any of this conversation's turns already in the JSONL
                remove_conv_from_live(conv_idx)
                record = None
                break                          # abandon this conversation

            # not truncated -> extract routing from THIS generation and save
            tensor_file, routing_meta = extract_and_pool_routing(prompt_len, conv_idx, turn_idx + 1)
            logit_capture.clear()

            if not response_text:
                print("  !! EMPTY GENERATION — check the chat template / terminators.")

            # Feed the model's OWN reply back so turn N+1 is correctly conditioned.
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

        if record is not None:
            if record["status"] == "running":
                record["status"] = "complete"
            record["final_history"] = history
            flush_results(results_log)

    if _TRUNCATED_TURNS:
        _tc = sorted(set(c for c, t, g in _TRUNCATED_TURNS))
        print("\n" + "=" * 60)
        print(f"SKIPPED {len(_tc)} conversations due to truncation "
              f"at MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
        print(f"  skipped conversation ids: {_tc}")
        print(f"  (absent from tensors, JSON, and JSONL — build_dataset globs)")
        print("=" * 60)
    else:
        print(f"\n[OK] No truncation at MAX_NEW_TOKENS={MAX_NEW_TOKENS} — all kept.")
    print(f"\nComplete.\n  Full results : {OUTPUT_RESULTS_FILE}\n  Live stream  : {LIVE_LOG_FILE}")


if __name__ == "__main__":
    run_dataset_benchmark()