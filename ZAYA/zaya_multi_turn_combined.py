import os
import re
import json
import ast
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

# full fresh run of all 300
#python zaya_multi_turn_combined.py

# resume from conversation 101 onward
#python zaya_multi_turn_combined.py --start 101 --limit 300

# just the first 50 (testing)
#python zaya_multi_turn_combined.py --limit 50

KMAX = 30   # keep the top-K token weights per (layer, expert); any K<=KMAX is a
            # feature-build knob later. Adds ONE payload key: 'topk_token_probs'.


def _pool_topk(token_probs, KMAX=KMAX):
    """(T, E) softmax routing weights -> (KMAX, E): the KMAX highest token weights
    for each expert, sorted descending, zero-padded when T < KMAX. Enables
    max (K=1) and top-K mean (topk[:K].mean(0)) pooling without re-running."""
    T, E = token_probs.shape
    k = min(KMAX, T)
    vals = token_probs.topk(k, dim=0).values
    if k < KMAX:
        vals = torch.cat([vals, token_probs.new_zeros(KMAX - k, E)], dim=0)
    return vals


# ==========================================
# PHASE 1: CONFIGURATION & MODEL LOADING
# ==========================================
MODEL_ID = "Zyphra/ZAYA1-8B"

# Everything is anchored to the SCRIPT's folder, not the current working
# directory, so `python zaya.py` behaves the same no matter where you launch it.
BASE_DIR = Path(__file__).resolve().parent

# --- dataset: use ONE of these two ---
# (a) local file  -> point LOCAL_DATASET at it, leave DATASET_ID as None
# (b) hub repo id -> set DATASET_ID, leave LOCAL_DATASET as None
def _find_dataset(base):
    """Prefer a family-specific curated_dataset_*.csv sitting next to this script
    (e.g. curated_dataset_pythonize.csv); fall back to plain curated_dataset.csv."""
    for c in sorted(base.glob("curated_dataset_*.csv")) + [base / "curated_dataset.csv"]:
        if c.exists():
            return c
    raise FileNotFoundError(f"No curated_dataset*.csv found in {base}")


LOCAL_DATASET = _find_dataset(BASE_DIR)
DATASET_ID = None
DATASET_SPLIT = None       # None -> take whatever split exists

CONV_COLUMN = "prompt"   # the only column used

MEAN_LOG_DIR = BASE_DIR / "mean_pool_tensors" / "multi-turn" / "moe_routing_tensors"
TOPK_LOG_DIR = BASE_DIR / "topk_pool_tensors" / "multi-turn" / "moe_routing_tensors"
OUTPUT_RESULTS_FILE = BASE_DIR / "benchmark_generation_outputs.json"  # rewritten EVERY turn
LIVE_LOG_FILE = BASE_DIR / "live_turns.jsonl"                         # appended EVERY turn

MAX_TURNS = 5              # ceiling only; a row with fewer prompts just ends early
MAX_NEW_TOKENS = 1024
# if a turn truncates (hits cap with no </think>), retry at these escalating budgets:
RETRY_BUDGETS = [2048, 4096]   # tried in order; [] disables auto-retry

# tracks turns that hit the token cap without closing </think> (still truncated)
_TRUNCATED_TURNS = []
TEST_LIMIT = 300

# ---- resumable extraction ----------------------------------------------------
# python zaya_multi_turn_topk.py --start 16 --limit 300
# --start is 1-based. A conversation whose first-turn tensor already exists is
# skipped, so a re-run continues instead of restarting from conversation 1.
import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument("--start", type=int, default=1)
_ap.add_argument("--limit", type=int, default=None)
_ap.add_argument("--ids", type=str, default=None,
                 help="comma-separated conversation_ids (0-indexed) to (re)extract, "
                      "overrides --start/--limit, overwrites existing tensors")
_args, _ = _ap.parse_known_args()
START_INDEX = max(1, _args.start)
SELECTED_IDS = None
if _args.ids:
    SELECTED_IDS = set(int(x) for x in _args.ids.replace(" ", "").split(",") if x != "")
    print(f"[ids] targeting {len(SELECTED_IDS)} conversations: {sorted(SELECTED_IDS)[:10]}"
          f"{'...' if len(SELECTED_IDS) > 10 else ''}")
if _args.limit is not None:
    TEST_LIMIT = _args.limit
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
# PHASE 2: ROUTER HOOKS  (ZAYA-SPECIFIC)
# ==========================================
# The code the Hub actually loads exposes routers as:
#     model.layers.{n}.mlp.gate.router_mlp   ->  class ZayaRouterMLP
#         .norm (ZayaRMSNorm) .fc1 .fc2 .out_proj (Linear) .act_fn (GELU)
#
# (The standalone modeling_zaya.py reference file has a DIFFERENT layout --
#  zaya_block.router.router_mlp as an nn.Sequential -- so match on the CLASS NAME,
#  which is stable across both, not on the module path.)
#
# ZayaRouterMLP output = raw per-token expert logits. Shape is either
# (B, S, E) or (B*S, E); _to_2d() normalises both to (num_tokens, E).
#
# Expert counts may differ per layer, so we zero-pad before stacking.

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


def _make_logit_hook(layer_n):
    def hook(module, inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(t):
            return
        if not _shape_diag["printed"]:
            print(f"[diag] ZayaRouterMLP L{layer_n} raw output shape={tuple(t.shape)} dtype={t.dtype}")
            _shape_diag["printed"] = True
        t = _to_2d(t.detach().float().cpu())
        if t is None:
            return
        layer_num_experts[layer_n] = t.shape[-1]
        logit_capture.setdefault(layer_n, []).append(t)
    return hook


router_layers = []
for name, module in model.named_modules():
    if type(module).__name__ != "ZayaRouterMLP":
        continue
    m = re.search(r"layers\.(\d+)\.", name)
    layer_n = int(m.group(1)) if m else len(router_layers)
    module.register_forward_hook(_make_logit_hook(layer_n))
    router_layers.append(layer_n)

print(f"Registered {len(router_layers)} ZayaRouterMLP hooks on layers: {router_layers}")
if not router_layers:
    print("!! No hooks attached. Candidate module names:")
    for name, module in model.named_modules():
        if "router" in name or "gate" in name:
            print("   ", name, "->", type(module).__name__)
    raise RuntimeError("Router modules not found — fix the class-name match above.")


def extract_and_pool_routing(prompt_len, conversation_id, turn_id):
    """
    Concatenate per-step router logits, isolate the generated-response rows,
    mean-pool over the token axis, and save.

    Prefill contributes `prompt_len` rows; each decode step contributes 1 row.
    Rows [prompt_len:] are therefore the routing decisions taken while processing
    the newly generated tokens.
    """
    layers = sorted(logit_capture.keys())
    if not layers:
        return None, {"tensor_matrix_path": None}

    pooled_logits = {}   # mean-pooled raw logits
    pooled_probs = {}    # mean-pooled softmax probabilities (scale-free, better for comparing)
    hist = {}            # hard top-1 expert counts over the response tokens
    topk = {}            # (KMAX, E) per layer: top token weights per expert
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
        probs = torch.softmax(resp, dim=-1)                # (T, E)
        pooled_probs[layer_n] = probs.mean(dim=0)
        topk[layer_n] = _pool_topk(probs, KMAX)            # (KMAX, E) NEW
        # softmax is monotonic, so argmax(logits) == argmax(prob) == the top-1 expert
        # (exact unless MOD balancing biases are active, which only touch the skip expert)
        hist[layer_n] = torch.bincount(resp.argmax(dim=-1), minlength=E).tolist()

    # Expert counts can differ per layer -> zero-pad to E_max so the rows can stack.
    e_max = max(v.shape[0] for v in pooled_logits.values())

    def _stack(d):
        rows = []
        for layer_n in layers:
            v = d[layer_n]
            pad = e_max - v.shape[0]
            rows.append(torch.nn.functional.pad(v, (0, pad)) if pad else v)
        return torch.stack(rows)

    def _stack_topk(d):
        rows = []
        for layer_n in layers:
            v = d[layer_n]                          # (KMAX, E_layer)
            pad = e_max - v.shape[-1]
            rows.append(torch.nn.functional.pad(v, (0, pad)) if pad else v)
        return torch.stack(rows)                    # (L, KMAX, E_max)

    matrix = _stack(pooled_logits)          # (num_moe_layers, E_max)
    prob_matrix = _stack(pooled_probs)
    topk_matrix = _stack_topk(topk)          # (L, KMAX, E_max)
    mask = [[1] * layer_num_experts[l] + [0] * (e_max - layer_num_experts[l]) for l in layers]

    # shared base payload (identical for both poolings — same generation)
    base_payload = {
        "pooled_logits": matrix,
        "pooled_probs": prob_matrix,
        "layer_indices": layers,
        "num_experts_per_layer": [layer_num_experts[l] for l in layers],
        "valid_mask": torch.tensor(mask, dtype=torch.bool),
        "response_token_count": resp_tokens,
        "top1_expert_histogram": hist,
    }

    fname = f"conv_{conversation_id:04d}_turn_{turn_id:02d}.pt"

    # MEAN payload: base only (build_dataset reads pooled_probs for mean pooling)
    torch.save(base_payload, MEAN_LOG_DIR / fname)

    # TOPK payload: base + the top-K token weights (enables K=1..KMAX pooling)
    topk_payload = dict(base_payload)
    topk_payload["topk_token_probs"] = topk_matrix   # (L, KMAX, E)
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
    'prompt' -> list of {"role","content"} dicts.
    Handles: an already-decoded list, a JSON string, a Python-repr string,
    ShareGPT-style {"from","value"}, and odd capitalisation.

    Any system message found in the data is DISCARDED -- no system role is ever
    sent to the model.
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
        if role == "system":
            continue                      # never fed to the model
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

    `history` contains user and assistant messages only. No system message is ever
    added.
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


def append_live(record):
    with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
        if conv_idx >= TEST_LIMIT:
            break
        if SELECTED_IDS is not None:
            if conv_idx not in SELECTED_IDS:
                continue
            # targeted re-extraction: overwrite existing (that is the point)
        else:
            if (conv_idx + 1) < START_INDEX:
                continue
            _existing = MEAN_LOG_DIR / f"conv_{conv_idx:04d}_turn_01.pt"
            if _existing.exists():
                print(f"  [skip] conversation {conv_idx + 1}: {_existing.name} exists")
                continue

        messages = parse_conversation(sample.get(CONV_COLUMN))

        # Replay ONLY the user prompts; the model writes its own assistant turns.
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

            # --- generate with automatic escalation on truncation ---
            budgets = [MAX_NEW_TOKENS] + [b for b in RETRY_BUDGETS if b > MAX_NEW_TOKENS]
            response_text = ""
            new_ids = None
            tensor_file = routing_meta = None
            used_budget = MAX_NEW_TOKENS
            for attempt, budget in enumerate(budgets):
                # context guard for this budget
                if prompt_len + budget >= MAX_CTX:
                    print(f"  (budget {budget} would exceed context {MAX_CTX}; "
                          f"stopping escalation)")
                    break
                logit_capture.clear()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=budget,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        eos_token_id=terminators,
                        pad_token_id=pad_id,
                    )
                new_ids = outputs[0][prompt_len:]
                response_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                used_budget = budget
                gen_len = int(new_ids.shape[0])
                truncated_now = (gen_len >= budget - 1) and ("</think>" not in response_text)
                if not truncated_now:
                    # extract from THIS (successful) generation's captured logits
                    tensor_file, routing_meta = extract_and_pool_routing(
                        prompt_len, conv_idx, turn_idx + 1)
                    logit_capture.clear()
                    if attempt > 0:
                        print(f"  [retry OK] closed </think> at budget {budget} "
                              f"({gen_len} tokens)")
                    break
                # truncated at this budget -> escalate if there is a larger one
                if attempt < len(budgets) - 1:
                    print(f"  !! truncated at {gen_len}/{budget} tokens (no </think>) "
                          f"-- retrying at {budgets[attempt + 1]}")
                else:
                    # exhausted all budgets: accept the truncated result, extract, flag
                    tensor_file, routing_meta = extract_and_pool_routing(
                        prompt_len, conv_idx, turn_idx + 1)
                    logit_capture.clear()
                    print(f"  !! STILL truncated after max budget {budget} "
                          f"-- conv {conv_idx} turn {turn_idx + 1} accepted as-is")

            print(f"MODEL: {response_text[:300]}")
            if not response_text:
                print("  !! EMPTY GENERATION — check the chat template / terminators.")

            # Feed the model's OWN reply back so turn N+1 is correctly conditioned.
            # An empty assistant message would corrupt the template, so skip it.
            if response_text:
                history.append({"role": "assistant", "content": response_text})

            gen_len = int(new_ids.shape[0])
            # after escalation: truncated only if STILL no </think> at the largest budget used
            is_truncated = (gen_len >= used_budget - 1) and ("</think>" not in response_text)
            if is_truncated:
                _TRUNCATED_TURNS.append((conv_idx, turn_idx + 1, gen_len))

            turn_record = {
                "turn_id": turn_idx + 1,
                "user_prompt": user_prompt,
                "model_response": response_text,
                "prompt_token_len": prompt_len,
                "generated_token_len": gen_len,
                "max_new_tokens_used": used_budget,
                "truncated": is_truncated,
                "raw_prompt_fed_to_model": prompt_text,
                **routing_meta,
            }

            record["turns"].append(turn_record)
            record["running_history"] = list(history)
            flush_results(results_log)
            append_live({"conversation_id": conv_idx, **turn_record})

            print(f"  saved -> {tensor_file}")

        if record["status"] == "running":
            record["status"] = "complete"
        record["final_history"] = history
        flush_results(results_log)

    # --- truncation summary ---
    if _TRUNCATED_TURNS:
        trunc_convs = sorted(set(c for c, t, g in _TRUNCATED_TURNS))
        print("\n" + "=" * 60)
        print(f"WARNING: {len(_TRUNCATED_TURNS)} turns STILL truncated at "
              f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
        print(f"  affected conversations ({len(trunc_convs)}): {trunc_convs}")
        print(f"  -> re-run just these at a higher budget, e.g.:")
        print(f"     (raise MAX_NEW_TOKENS to 2048 and use --ids {','.join(map(str, trunc_convs))})")
        print("=" * 60)
    else:
        print(f"\n[OK] No truncation at MAX_NEW_TOKENS={MAX_NEW_TOKENS} — all responses closed </think>.")
    print(f"\nComplete.\n  Full results : {OUTPUT_RESULTS_FILE}\n  Live stream  : {LIVE_LOG_FILE}")


if __name__ == "__main__":
    run_dataset_benchmark()