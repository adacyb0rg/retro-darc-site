# Retro-DARC-Lite adapter — quickstart

Function-preserving depth-memory read for pretrained causal LMs.
At gate γ=0 (the initialization) the model's outputs are **bitwise identical**
to the base model — verified on gpt2, pythia-410m, and Qwen2.5-0.5B (paper §10, E10.1).
Detaching the hooks restores the base exactly.

## Install

```bash
pip install torch transformers
# plus this file: depth_adapter.py (no other project code needed)
```

Or install the module straight from the repository:

```bash
pip install "git+https://github.com/adacyb0rg/retro-darc"
```

## Attach to a pretrained model (exact no-op at insertion)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from depth_adapter import DepthReadAdapter, HookedDepthAdapter, get_blocks, MODEL_TABLE

model_id = "EleutherAI/pythia-410m"        # or gpt2 / Qwen2.5-0.5B / ... (see MODEL_TABLE)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

cfg = MODEL_TABLE[model_id]
ins = 2 * cfg["layers"] // 3               # insertion depth INS = 2L/3 (the paper's default)
adapter = DepthReadAdapter(d=cfg["d"], ins=ins, key_mode="delta", null=True)
hooked = HookedDepthAdapter(get_blocks(model, model_id), adapter)

tok = AutoTokenizer.from_pretrained(model_id)
x = tok("The capital of France is", return_tensors="pt").input_ids

base = model(x).logits                     # before insertion
hooked.attach()                            # γ = 0 → exact identity
after = model(x).logits
assert torch.equal(base, after)            # bitwise, float32 CPU

hooked.detach()                            # removes the hooks; base restored bitwise
```

## Train (adapter-only; the base stays frozen)

```python
hooked.attach()
for p in model.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3)   # 4·64·d + 65 trainable params
# ... standard LM loss over your continued-training tokens ...
```

The gate trains first (Prop. 2′): at step 0 only ∇γ is nonzero; everything else
follows once γ moves. Measured reference runs (frozen pythia-410m, wikitext-103,
matched ~262k params): see paper §10 rows E12/E15.

## The causal audit (what weight-touching adapters cannot express)

```python
adapter.intervention = "zero"              # bank zeroed -> frozen loss returns EXACTLY
adapter.intervention = "shuffle"           # content shuffled -> removes 70-107% of the gain
adapter.intervention = "randn_same_norm"   # same-norm noise -> lands WORSE than frozen
adapter.intervention = None                # back to normal reads
```

Zeroing the bank returned the frozen loss bit-exactly in all 16 CPU/MPS tests
across the paper's runs and to ≤1×10⁻⁷ on CUDA (kernel reduction-order noise).
Run this battery on your own model before trusting any gain.

## Pretrained adapter checkpoints (this kit)

| file | head | trained on |
|---|---|---|
| `pythia-410m_rd_lite_seed0.pt` | RD-Lite (delta keys + Softmax1 null) | frozen pythia-410m, wikitext-103, seed 0 |
| `pythia-410m_attnres_style_seed0.pt` | AttnRes-style (output keys + softmax) | same run, same budget |

```python
adapter.load_state_dict(torch.load("pythia-410m_rd_lite_seed0.pt"))
```

(They are plain `state_dict`s for the `DepthReadAdapter` above with
`d=1024, ins=16`; seed-1 CUDA-replication checkpoints are in the repository
under `spark_bundle/adapters_ckpt/`.)

## Two measured design rules (from the paper's failed predictions)

1. **ℓ2-normalize each delta before the key projection** (E13) — raw keys are
   ill-conditioned near the top of the stack; normalization holds full-bank
   κ at 3.0–3.7 on all five models tested.
2. **Initialize every gate at zero — including from-scratch builds** (E14):
   γ₀=1 cost +0.0142 CE on 3/3 seeds; the deficit is pure initialization shock.

Full paper, raw JSONs, and the released harness:
https://adacyb0rg.github.io/retro-darc/
