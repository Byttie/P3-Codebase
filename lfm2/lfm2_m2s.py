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
# LFM2.5-8B-A1B loads natively as the `lfm2moe` architecture (Lfm2MoeForCausalLM)
# in recent transformers. If from_pretrained raises "model type `lfm2moe` not
# recognized", run `pip install -U transformers`.
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

CONV_COLUMN = "prompt"     # the only column used; each cell is one user message

ROUTING_LOG_DIR = BASE_DIR / "moe_routing_tensors"
OUTPUT_RESULTS_FILE = BASE_DIR / "benchmark_generation_outputs.json"  # rewritten each row
LIVE_LOG_FILE = BASE_DIR / "live_prompts.jsonl"                       # appended each row

MAX_NEW_TOKENS = 1024      # NOTE: LFM2.5 is a reasoning model (chain-of-thought before
                           # the final answer), so you may want to raise this.
TEST_LIMIT = 50
SEED = 1234

ROUTING_LOG_DIR.mkdir(parents=True, exist_ok=True)
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
    # (8.3B in 8-bit ~= 9 GB). To switch, replace the block above with:
    #     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
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
# Structure (from the transformers source + the model's own module tree):
#   model.layers.{n}.feed_forward   -> Lfm2MoeSparseMoeBlock  (MoE layers only; the
#                                      first `num_dense_layers` layers are dense
#                                      Lfm2MoeMLP -- which is why hooks start at layer 2)
#       .gate     -> the router. Either a plain nn.Linear (older versions) or an
#                    Lfm2MoeTopKRouter WRAPPER (newer versions).
#       .experts  -> Lfm2MoeExperts (fused/batched experts)
#
# The router projects hidden_size -> num_experts (=32) to per-token logits, then routes
# with SIGMOID scores + top-k (k = num_experts_per_tok = 4). Two consequences vs ZAYA:
#
#   1) The Lfm2MoeTopKRouter wrapper's forward returns ONLY the top-k (indices,
#      weights) -- NOT the full 32-wide logits. So when the module output lacks a
#      32-wide tensor, we RECOMPUTE the full logits from the router INPUT and its gate
#      weight:  logits = input @ W_gate.T . (When the gate is a plain nn.Linear we just
#      use its output.) Either way `extract_and_pool_routing` gets full logits.
#
#   2) LFM2 gates with sigmoid, not softmax -- so pooled "probs" are sigmoid and the
#      histogram counts the top-k selected experts rather than a single argmax.

NUM_EXPERTS = (
    getattr(model.config, "num_experts", None)
    or getattr(model.config, "num_local_experts", None)
    or getattr(model.config, "n_routed_experts", None)
)
TOP_K = getattr(model.config, "num_experts_per_tok", 4)
print(f"[diag] num_experts={NUM_EXPERTS}  top_k={TOP_K}  "
      f"num_dense_layers={getattr(model.config, 'num_dense_layers', '?')}")

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

    Returns None for a bnb-quantized Linear (its weight is packed uint8, not a
    floating (E, hidden) matrix), in which case the hook uses the module's output.
    """
    for _, p in module.named_parameters(recurse=True):
        if p.dim() == 2 and p.shape[0] == E and p.is_floating_point():
            return p
    return None


def _make_router_hook(layer_n, gate_weight):
    def hook(module, inputs, output):
        # Preferred: the module already emits full per-expert logits (nn.Linear gate).
        t = _logits_from_output(output, NUM_EXPERTS)
        # Fallback: wrapper router (Lfm2MoeTopKRouter) returns only top-k -> recompute
        # the full logits from the router INPUT and its gate weight.
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
    # The router lives at ...feed_forward.gate in every MoE layer.
    if not name.endswith("feed_forward.gate"):
        continue
    m = re.search(r"layers\.(\d+)\.", name)
    layer_n = int(m.group(1)) if m else len(router_layers)
    # Full-precision gate weight for the wrapper/quantized path (None for a plain fp
    # nn.Linear, where the hook uses the module's own float output instead).
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


def extract_and_pool_routing(prompt_len, row_id):
    """
    Concatenate per-step router logits, isolate the GENERATED-response rows,
    mean-pool over the token axis, and save.

    Prefill contributes `prompt_len` rows; each decode step contributes 1 row.
    Rows [prompt_len:] are therefore the routing decisions on the generated tokens.

    LFM2-MoE gates with SIGMOID and activates TOP_K experts/token, so:
      * pooled_probs   = mean over tokens of sigmoid(logits)   (independent per-expert
                         gate scores; they do NOT sum to 1)
      * topk_histogram = how often each expert lands in a token's top-K. Uses the
                         sigmoid-score top-K and ignores the optional `expert_bias`
                         additive selection term, so it is a faithful approximation
                         of the routed set, not a byte-exact replica.
    """
    layers = sorted(logit_capture.keys())
    if not layers:
        return None, {"tensor_matrix_path": None}

    pooled_logits = {}   # mean-pooled raw logits
    pooled_probs = {}    # mean-pooled sigmoid gate scores
    hist = {}            # top-K expert selection counts over the response tokens
    resp_tokens = 0

    for layer_n in layers:
        logits = torch.cat(logit_capture[layer_n], dim=0)     # (num_tokens, E_layer)
        E = logits.shape[-1]
        if logits.shape[0] <= prompt_len:
            pooled_logits[layer_n] = torch.zeros(E)
            pooled_probs[layer_n] = torch.zeros(E)
            hist[layer_n] = [0] * E
            continue
        resp = logits[prompt_len:, :]
        resp_tokens = resp.shape[0]
        pooled_logits[layer_n] = resp.mean(dim=0)
        pooled_probs[layer_n] = torch.sigmoid(resp).mean(dim=0)         # sigmoid, not softmax
        k = min(TOP_K, E)
        topk_idx = resp.topk(k, dim=-1).indices.reshape(-1)            # top-K per token
        hist[layer_n] = torch.bincount(topk_idx, minlength=E).tolist()

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

    payload = {
        "pooled_logits": matrix,
        "pooled_probs": prob_matrix,           # sigmoid gate scores
        "layer_indices": layers,
        "num_experts_per_layer": [layer_num_experts[l] for l in layers],
        "valid_mask": torch.tensor(mask, dtype=torch.bool),
        "response_token_count": resp_tokens,
        "top_k": TOP_K,
        "topk_expert_histogram": hist,
    }

    fname = f"prompt_{row_id:04d}.pt"
    torch.save(payload, ROUTING_LOG_DIR / fname)

    meta = {
        "tensor_matrix_path": fname,
        "routing_matrix_shape": list(matrix.shape),
        "routed_response_tokens": resp_tokens,
        "moe_layer_indices": layers,
    }
    return fname, meta


# ==========================================
# PHASE 3: PROMPT EXTRACTION
# ==========================================
def extract_prompt(raw):
    """
    The 'prompt' cell holds ONE user message as a dict, e.g.
        {"role": "user", "content": "what is 1+1"}

    Returns the content string, or None if unreadable. Also tolerates a JSON/repr
    string wrapping that dict, a single-element list, a bare string, or
    ShareGPT-style {"value"} / {"text"} keys.
    """
    if raw is None:
        return None
    obj = raw
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                return s          # the cell is just the prompt text itself
    if isinstance(obj, list):
        obj = obj[0] if obj else None
    if isinstance(obj, dict):
        content = obj.get("content", obj.get("value", obj.get("text", "")))
        content = str(content).strip()
        return content or None
    if isinstance(obj, str):
        return obj.strip() or None
    return None


def build_prompt(user_content):
    """
    Wrap a single user message in the model's chat template. Tokenise with
    add_special_tokens=False: the template already carries BOS/role markers, and a
    second auto-BOS would shift prompt_len out of sync with the router rows.
    """
    msgs = [{"role": "user", "content": user_content}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text = tokenizer.bos_token or ""
    text += f"<|im_start|>user\n{user_content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


# ==========================================
# PHASE 4: INCREMENTAL WRITERS
# ==========================================
def flush_results(results_log):
    """Atomic rewrite, so the file is never half-written when opened mid-run."""
    tmp = OUTPUT_RESULTS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUTPUT_RESULTS_FILE)


def append_live(record):
    with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ==========================================
# PHASE 5: BENCHMARK  (one prompt -> one generation)
# ==========================================
def run_dataset_benchmark():
    print("Loading dataset...")
    if LOCAL_DATASET is not None:
        path = Path(LOCAL_DATASET).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
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
        split_name = list(loaded.keys())[0]
        dataset = loaded[split_name]
        print(f"Using split '{split_name}' ({len(dataset)} rows)")

    if CONV_COLUMN not in dataset.column_names:
        raise KeyError(f"Column {CONV_COLUMN!r} not found. Available: {dataset.column_names}")

    open(LIVE_LOG_FILE, "w").close()
    results_log = []

    for row_idx, sample in enumerate(dataset):
        if row_idx >= TEST_LIMIT:
            break

        user_prompt = extract_prompt(sample.get(CONV_COLUMN))
        print(f"\n{'=' * 70}\nPROMPT {row_idx + 1}/{TEST_LIMIT}")

        if not user_prompt:
            print("  Warning: empty/unreadable prompt cell. Skipping.")
            continue

        prompt_text = build_prompt(user_prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        if prompt_len + MAX_NEW_TOKENS >= MAX_CTX:
            print(f"  Prompt too long ({prompt_len} tok) for the context window. Skipping.")
            continue

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

        tensor_file, routing_meta = extract_and_pool_routing(prompt_len, row_idx)
        logit_capture.clear()

        print(f"MODEL: {response_text[:300]}")
        if not response_text:
            print("  !! EMPTY GENERATION — check the chat template / terminators.")

        record = {
            "prompt_id": row_idx,
            "user_prompt": user_prompt,
            "model_response": response_text,
            "prompt_token_len": prompt_len,
            "generated_token_len": int(new_ids.shape[0]),
            "raw_prompt_fed_to_model": prompt_text,
            **routing_meta,
        }
        results_log.append(record)
        flush_results(results_log)
        append_live(record)

        print(f"  saved -> {tensor_file}")

    print(f"\nComplete.\n  Results : {OUTPUT_RESULTS_FILE}\n  Live    : {LIVE_LOG_FILE}")


if __name__ == "__main__":
    run_dataset_benchmark()