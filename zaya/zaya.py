import os
import re
import json
import ast
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

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
LOCAL_DATASET = BASE_DIR / "curated_dataset.csv"
DATASET_ID = None
DATASET_SPLIT = None       # None -> take whatever split exists

CONV_COLUMN = "prompt"   # the only column used

# Used only if a row has no system message of its own.
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI Assistant!"

ROUTING_LOG_DIR = BASE_DIR / "moe_routing_tensors"
OUTPUT_RESULTS_FILE = BASE_DIR / "benchmark_generation_outputs.json"  # rewritten EVERY turn
LIVE_LOG_FILE = BASE_DIR / "live_turns.jsonl"                         # appended EVERY turn

MAX_TURNS = 5              # ceiling only; a row with fewer prompts just ends early
MAX_NEW_TOKENS = 1024
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
        pooled_probs[layer_n] = torch.softmax(resp, dim=-1).mean(dim=0)
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

    matrix = _stack(pooled_logits)          # (num_moe_layers, E_max)
    prob_matrix = _stack(pooled_probs)
    mask = [[1] * layer_num_experts[l] + [0] * (e_max - layer_num_experts[l]) for l in layers]

    payload = {
        "pooled_logits": matrix,
        "pooled_probs": prob_matrix,
        "layer_indices": layers,
        "num_experts_per_layer": [layer_num_experts[l] for l in layers],
        "valid_mask": torch.tensor(mask, dtype=torch.bool),
        "response_token_count": resp_tokens,
        "top1_expert_histogram": hist,
    }

    fname = f"conv_{conversation_id:04d}_turn_{turn_id:02d}.pt"
    torch.save(payload, ROUTING_LOG_DIR / fname)

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

    `history` may begin with a {"role": "system", ...} message. Some chat templates
    refuse a system role outright -- if that happens we fold the system text into
    the first user message rather than silently dropping it.
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        except Exception:
            if history and history[0]["role"] == "system":
                sys_txt = history[0]["content"]
                folded = [dict(m) for m in history[1:]]
                if folded and folded[0]["role"] == "user":
                    folded[0]["content"] = f"{sys_txt}\n\n{folded[0]['content']}"
                return tokenizer.apply_chat_template(
                    folded, tokenize=False, add_generation_prompt=True
                )
            raise

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

    open(LIVE_LOG_FILE, "w").close()
    results_log = []

    for conv_idx, sample in enumerate(dataset):
        if conv_idx >= TEST_LIMIT:
            break

        messages = parse_conversation(sample.get(CONV_COLUMN))

        # Replay ONLY the user prompts; the model writes its own assistant turns.
        user_turns = [m["content"] for m in messages if m["role"] == "user"]

        # The row's own system message, if it has one; otherwise the default.
        sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
        system_prompt = sys_msgs[0] if sys_msgs else DEFAULT_SYSTEM_PROMPT

        print(f"\n{'=' * 70}\nCONVERSATION {conv_idx + 1}/{TEST_LIMIT}")
        print(f"parsed {len(messages)} messages -> {len(user_turns)} user turns")
        print(f"system: {system_prompt[:80]}{' (default)' if not sys_msgs else ''}")

        if not user_turns:
            print("  Warning: no user prompts in this row. Skipping.")
            continue

        record = {
            "conversation_id": conv_idx,
            "system_prompt": system_prompt,
            "system_prompt_from_dataset": bool(sys_msgs),
            "dataset_user_turns": len(user_turns),
            "status": "running",
            "turns": [],
        }
        results_log.append(record)
        flush_results(results_log)          # visible before the first token is generated

        # System message is turn 0 of the history and stays there for every turn.
        history = [{"role": "system", "content": system_prompt}]

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

            tensor_file, routing_meta = extract_and_pool_routing(prompt_len, conv_idx, turn_idx + 1)
            logit_capture.clear()

            print(f"MODEL: {response_text[:300]}")
            if not response_text:
                print("  !! EMPTY GENERATION — check the chat template / terminators.")

            # Feed the model's OWN reply back so turn N+1 is correctly conditioned.
            # An empty assistant message would corrupt the template, so skip it.
            if response_text:
                history.append({"role": "assistant", "content": response_text})

            turn_record = {
                "turn_id": turn_idx + 1,
                "user_prompt": user_prompt,
                "model_response": response_text,
                "prompt_token_len": prompt_len,
                "generated_token_len": int(new_ids.shape[0]),
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

    print(f"\nComplete.\n  Full results : {OUTPUT_RESULTS_FILE}\n  Live stream  : {LIVE_LOG_FILE}")


if __name__ == "__main__":
    run_dataset_benchmark()