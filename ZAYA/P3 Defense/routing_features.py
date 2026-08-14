"""
routing_features.py  (v2 — local pooling modes)
-----------------------------------------------
Turn ONE routing payload into a fixed-length feature vector.

LOCAL POOLING (token axis, within a prompt) happens here, controlled by
FeatureConfig.local_pool:

  'mean'  -> use pooled_probs (the mean your extraction already saved).
  'topk'  -> use topk_token_probs[:, :K, :].mean(1)  (top-K mean pooling,
             the anti-dilution pooling from your notes). REQUIRES that you
             re-extracted with topk_extraction_patch.py so the payload carries
             'topk_token_probs' of shape (L, Kmax, E). If that field is absent
             the code falls back to mean and warns once.

Existing files (used for your PCA) only have the mean, so 'topk' silently
degrades to 'mean' on them — that is expected, and is why top-K needs a
re-extraction pass.
"""

from dataclasses import dataclass
import torch

_WARNED = {"no_topk": False}


@dataclass
class FeatureConfig:
    use_probs: bool = True
    use_top1_freq: bool = True
    use_logits: bool = False
    local_pool: str = "mean"     # 'mean' | 'topk'
    K: int = 5                    # top-K over tokens when local_pool='topk'
    drop_empty: bool = True


def _local_probs(payload, cfg):
    """Return the (L, E) per-expert routing matrix after local pooling."""
    if cfg.local_pool == "topk" and "topk_token_probs" in payload:
        t = payload["topk_token_probs"].float()      # (L, Kmax, E), sorted desc over tokens
        K = min(cfg.K, t.shape[1])
        return t[:, :K, :].mean(dim=1)                # (L, E)
    if cfg.local_pool == "topk" and not _WARNED["no_topk"]:
        print("[routing_features] local_pool='topk' but payload has no "
              "'topk_token_probs' -> falling back to mean. Re-extract with "
              "topk_extraction_patch.py to enable top-K.")
        _WARNED["no_topk"] = True
    return payload["pooled_probs"].float()            # (L, E) mean


def _top1_freq(payload, L, E):
    rtc = max(1, int(payload.get("response_token_count", 0)))
    layers = list(payload["layer_indices"])
    hist = payload["top1_expert_histogram"]
    out = torch.zeros(L, E)
    for row, l in enumerate(layers):
        h = torch.as_tensor(hist[l], dtype=torch.float32)
        out[row, : h.numel()] = h / rtc
    return out


def payload_to_vector(payload, cfg: FeatureConfig):
    mask = payload["valid_mask"].bool()
    L, E = mask.shape
    rtc = int(payload.get("response_token_count", 0))
    ok = not (cfg.drop_empty and rtc == 0)

    parts = []
    if cfg.use_probs:
        parts.append((_local_probs(payload, cfg) * mask).reshape(-1))
    if cfg.use_top1_freq:
        parts.append((_top1_freq(payload, L, E) * mask).reshape(-1))
    if cfg.use_logits:
        parts.append((payload["pooled_logits"].float() * mask).reshape(-1))

    vec = torch.cat(parts) if parts else torch.zeros(0)
    return vec, ok, {"L": L, "E": E, "response_tokens": rtc, "dim": vec.numel()}
