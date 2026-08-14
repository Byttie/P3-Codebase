"""
topk_extraction_patch.py
------------------------
Your .pt files already mean-pooled over tokens, so the token axis is gone and
top-K local pooling CANNOT be recovered from them. To enable top-K you must run
ONE more extraction pass that keeps the top-K token values per (layer, expert).

Drop `pool_layer_topk` into zaya_m2s.py / zaya_multi_turn.py and add its output
to the saved payload. You keep everything you already save; you just ADD one
field, `topk_token_probs`, of shape (L, KMAX, E). Then any K <= KMAX becomes a
train-time knob (no re-running the model per K).

--- how to wire it into extract_and_pool_routing (both scripts) ---

    KMAX = 10
    topk = {}                       # add alongside pooled_logits / pooled_probs
    for layer_n in layers:
        ...
        resp = logits[prompt_len:, :]              # (T, E) you already have
        pooled_logits[layer_n] = resp.mean(dim=0)
        probs = torch.softmax(resp, dim=-1)        # (T, E)
        pooled_probs[layer_n]  = probs.mean(dim=0) # unchanged
        topk[layer_n] = pool_layer_topk(probs, KMAX)   # (KMAX, E)  <-- NEW

    # stack (mirror your _stack padding to e_max on the EXPERT axis):
    def _stack_topk(d):
        rows = []
        for layer_n in layers:
            v = d[layer_n]                          # (KMAX, E_layer)
            pad = e_max - v.shape[-1]
            rows.append(torch.nn.functional.pad(v, (0, pad)) if pad else v)
        return torch.stack(rows)                    # (L, KMAX, e_max)

    payload["topk_token_probs"] = _stack_topk(topk)     # <-- ADD THIS KEY
"""

import torch


def pool_layer_topk(token_probs, KMAX=10):
    """
    token_probs: (T, E) softmax routing weights for the response tokens.
    Returns (KMAX, E): for each expert (column), the KMAX highest token weights,
    sorted descending, zero-padded if T < KMAX. Top-K mean for any K<=KMAX is
    then topk[:K].mean(0) at feature-build time.
    """
    T, E = token_probs.shape
    k = min(KMAX, T)
    vals = token_probs.topk(k, dim=0).values          # (k, E) desc over tokens
    if k < KMAX:
        vals = torch.cat([vals, token_probs.new_zeros(KMAX - k, E)], dim=0)
    return vals
