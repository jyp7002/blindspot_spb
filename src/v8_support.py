"""Support dumps for SUP1 (experiments_v8).

WHY THIS EXISTS: experiments_v8 §SUP1 says its inputs are "existing top-1%
supports per (cell, seed)" with "GPU ~ 0". That was not true when v8 was
written -- nothing in the pipeline ever persisted an edit's support. The panels
store a scalar removal plus alpha traces; the only np.save in colab_t2t4.py
writes t4 bias profiles. Regenerating a support means re-running the two task
vector trainings, which run_ablate_attn.py documents as the dominant cost.

So DEC dumps the support as it builds it, and SUP1 becomes analysis-only as
intended. One .npz per (panel, target, axis, seed, designer), written once per
training unit -- the C-ref support, which is the object SUP1 characterises.

FORMAT (npz, compressed):
  keys              : (M,) unicode module names, canonical sorted order
  shape__<i>        : (2,) int64 tensor shape for module i
  idx__<i>          : (k_i,) int32 flat indices into that module, sorted ascending
  sgn__<i>          : (k_i,) int8 sign in {-1,+1} at those coordinates
  meta              : json blob (target, axis, seed, designer, density, n_params...)

Indices are flat within their own module, so a module's coordinates are
comparable across cells only when the shapes match -- which SUP1's
shape-compatibility rule already requires.
"""
import os, json
import numpy as np

SUPPORT_DIRNAME = "supports"


def support_path(results_root, panel, target, axis, seed, designer):
    d = os.path.join(results_root, panel, SUPPORT_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{target}|{axis}|s{seed}|{designer}.npz")


def dump_support(path, E_or_v, mask_fn=None, meta=None):
    """Persist the nonzero support + signs of an edit dict {key: tensor}.

    E_or_v: the *constructed edit* E (already sparse). Nonzeros define the
    support; their signs define the sign field. Passing the dense contrast v
    with a mask_fn is also supported for callers that build the mask first.
    """
    import torch
    keys = sorted(E_or_v.keys())
    out = {"keys": np.array(keys, dtype=object)}
    total_k, total_n = 0, 0
    for i, k in enumerate(keys):
        t = E_or_v[k]
        m = (t != 0) if mask_fn is None else mask_fn(k, t)
        flat_idx = torch.nonzero(m.flatten(), as_tuple=False).flatten()
        vals = t.flatten()[flat_idx]
        out[f"shape__{i}"] = np.array(tuple(t.shape), dtype=np.int64)
        out[f"idx__{i}"] = flat_idx.cpu().numpy().astype(np.int32)
        out[f"sgn__{i}"] = torch.sign(vals).cpu().numpy().astype(np.int8)
        total_k += int(flat_idx.numel())
        total_n += int(t.numel())
    md = dict(meta or {})
    md.update(n_support=total_k, n_params=total_n,
              density=total_k / max(total_n, 1), n_modules=len(keys))
    out["meta"] = np.array(json.dumps(md), dtype=object)
    np.savez_compressed(path, **out)
    return md


def load_support(path):
    """-> (meta dict, {module_key: (shape, idx int32, sgn int8)})"""
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    meta = json.loads(str(z["meta"]))
    mods = {}
    for i, k in enumerate(keys):
        mods[k] = (tuple(int(x) for x in z[f"shape__{i}"]),
                   z[f"idx__{i}"], z[f"sgn__{i}"])
    return meta, mods


# ---- module-name parsing (llama/qwen/gemma/phi all use HF's layer naming) ----

def parse_module(key):
    """'model.layers.12.self_attn.q_proj' -> (12, 'q_proj'). (None, key) if odd."""
    parts = key.split(".")
    layer, proj = None, parts[-1]
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer = int(parts[i + 1])
            break
    return layer, proj
