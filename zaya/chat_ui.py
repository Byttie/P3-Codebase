import os
import csv
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Zyphra/ZAYA1-8B"
print(f"Loading {model_id} strictly to GPU...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    ),
    trust_remote_code=True
)

terminators = [tokenizer.eos_token_id]
for special_token in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>"]:
    if special_token in tokenizer.vocab:
        terminators.append(tokenizer.vocab[special_token])

# Router logits aren't exposed via model outputs in this build, so we hook
# each layer's ZayaRouterMLP directly. Its output is the raw per-expert
# logits (pre-softmax), matching the ZAYA1 router equation in the paper:
# s_l = softmax(MLP(RMSNorm(r_l))). Confirmed structure per layer:
# model.layers.{i}.mlp.gate (ZayaRouter) -> .router_mlp (ZayaRouterMLP).
ZAYA_USES_MOD = bool(getattr(model.config, "zaya_use_mod", False))
ROUTING_LOG_DIR = "moe_routing_logs"
os.makedirs(ROUTING_LOG_DIR, exist_ok=True)
turn_counter = 0

router_capture = {}  # layer_idx -> chronological list of tensors


def _make_router_hook(layer_idx):
    def hook(module, inputs, output):
        router_capture.setdefault(layer_idx, []).append(output.detach().float().cpu())
    return hook


hook_count = 0
for name, module in model.named_modules():
    if type(module).__name__ == "ZayaRouterMLP":
        module.register_forward_hook(_make_router_hook(hook_count))
        hook_count += 1

print(f"✅ Registered {hook_count} router hooks.")


def log_moe_routing(full_ids, prompt_len, turn_id):
    """Reassembles per-token, per-layer router logits captured via hooks
    during generate(). Note: the very last generated token never gets a
    router entry, since generation stops right after it's sampled and no
    further forward pass runs to compute its routing."""
    seq_len = full_ids.shape[1]
    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    layer_probs = {}
    for layer_idx, chunks in router_capture.items():
        logits = torch.cat(chunks, dim=1).squeeze(0)  # (captured_len, num_experts)
        layer_probs[layer_idx] = torch.softmax(logits, dim=-1)

    captured_len = next(iter(layer_probs.values())).shape[0] if layer_probs else 0
    usable_end = min(seq_len, captured_len)

    print(f"🔍 router_capture has {len(layer_probs)} layers, "
          f"{captured_len} captured positions (sequence length is {seq_len})")

    def write_csv(path, start, end):
        end = min(end, usable_end)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["token_index", "token_id", "token", "layer", "num_experts",
                 "expert_id", "expert_prob", "is_top_choice", "is_skip_expert"]
            )
            for layer_idx in sorted(layer_probs.keys()):
                probs = layer_probs[layer_idx]
                num_experts = probs.shape[-1]
                skip_idx = num_experts - 1 if ZAYA_USES_MOD else None
                for pos in range(start, end):
                    p = probs[pos]
                    top_expert = int(torch.argmax(p).item())
                    for expert_id in range(num_experts):
                        writer.writerow([
                            pos, full_ids[0, pos].item(), tokens[pos], layer_idx, num_experts,
                            expert_id, round(p[expert_id].item(), 6),
                            expert_id == top_expert, expert_id == skip_idx
                        ])

    prompt_csv = os.path.join(ROUTING_LOG_DIR, f"turn_{turn_id:03d}_prompt.csv")
    response_csv = os.path.join(ROUTING_LOG_DIR, f"turn_{turn_id:03d}_response.csv")
    write_csv(prompt_csv, 0, prompt_len)
    write_csv(response_csv, prompt_len, seq_len)
    print(f"📊 Routing logs saved: {prompt_csv} | {response_csv}")


def generate_response(message, history):
    formatted_messages = []

    def get_text(msg):
        if isinstance(msg, str): return msg
        if isinstance(msg, dict): return msg.get("text", msg.get("content", str(msg)))
        if isinstance(msg, (list, tuple)) and len(msg) > 0 and isinstance(msg[0], dict):
            return msg[0].get("text", str(msg))
        return str(msg)

    if history:
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item, dict):
                formatted_messages.append({"role": "user", "content": get_text(item[0])})
                formatted_messages.append({"role": "assistant", "content": get_text(item[1])})
            elif isinstance(item, dict) and "role" in item:
                formatted_messages.append({"role": item["role"], "content": get_text(item["content"])})
            elif hasattr(item, "role") and hasattr(item, "content"):
                formatted_messages.append({"role": item.role, "content": get_text(item.content)})

    formatted_messages.append({"role": "user", "content": get_text(message)})

    prompt = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[1]
    print(f"\n🧠 Generating response...")

    router_capture.clear()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )

    response_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    global turn_counter
    turn_counter += 1
    log_moe_routing(outputs, prompt_len, turn_counter)

    print("✅ Done!")
    return response_text


print("Starting Gradio interface...")
demo = gr.ChatInterface(
    fn=generate_response,
    title="ZAYA1-8B Local Hub",
    description="Running locally on RTX 5070 Ti"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)