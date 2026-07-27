"""Shared hook machinery for Tasks 1 and 3 (claude_code_plan.md, "Shared reference").

Implements the contract exactly:
  - forward pre-hooks (with_kwargs=True) on blocks[0..INS] record the incoming hidden
    state; the hook on blocks[INS] builds the per-token bank Mem from the recorded
    stream (delta keys: stream[i+1]-stream[i]; output keys: stream[1..INS]) and
    returns ((h + gamma * Wo_out, *args[1:]), kwargs).
  - read: q = Uq(h) [d->64, bias]; scores = einsum('btr,btsr->bts', q, Pk(Mem))/8
    [Pk: d->64, no bias]; Softmax1 (null-reserving) for RD-Lite, plain softmax for
    AttnRes-style; r = einsum('bts,btsd->btd', a, Mem); Wo_out = Vo(Uo(r))
    [Uo: d->64 no bias, Vo: 64->d no bias, Vo init normal std 0.02];
    gamma = nn.Parameter(zeros(())).
  - trainable params = 4*64*d + 64 (Uq bias) + 1 (gamma).
  - eval-only interventions replace Mem with zeros / batch-permutation / gaussian
    rescaled per-vector to Mem's norms.
"""
import math

import torch
import torch.nn as nn

RANK = 64


class DepthReadAdapter(nn.Module):
    """Token-conditioned depth read over the layer stream entering blocks 0..INS."""

    def __init__(self, d, ins, key_mode="delta", null=True, rank=RANK):
        super().__init__()
        assert key_mode in ("delta", "output")
        self.d, self.ins, self.rank = d, ins, rank
        self.key_mode, self.null = key_mode, null
        self.Uq = nn.Linear(d, rank)
        self.Pk = nn.Linear(d, rank, bias=False)
        self.Uo = nn.Linear(d, rank, bias=False)
        self.Vo = nn.Linear(rank, d, bias=False)
        nn.init.normal_(self.Vo.weight, std=0.02)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.intervention = None  # None | 'zero' | 'shuffle' | 'randn_same_norm'

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def build_mem(self, stream):
        if self.key_mode == "delta":
            return torch.stack([stream[i + 1] - stream[i] for i in range(self.ins)], dim=2)
        return torch.stack(stream[1:self.ins + 1], dim=2)

    def read(self, h, mem):
        if self.intervention == "zero":
            mem = torch.zeros_like(mem)
        elif self.intervention == "shuffle":
            mem = mem[torch.randperm(mem.shape[0], device=mem.device)]
        elif self.intervention == "randn_same_norm":
            r = torch.randn_like(mem)
            mem = r * mem.norm(dim=-1, keepdim=True) / (r.norm(dim=-1, keepdim=True) + 1e-9)
        q = self.Uq(h)
        scores = torch.einsum("btr,btsr->bts", q, self.Pk(mem)) / math.sqrt(self.rank)
        if self.null:
            m = scores.max(-1, keepdim=True).values
            a = torch.exp(scores - m)
            a = a / (torch.exp(-m) + a.sum(-1, keepdim=True))
        else:
            a = torch.softmax(scores, -1)
        r = torch.einsum("bts,btsd->btd", a, mem)
        return self.Vo(self.Uo(r))


class HookedDepthAdapter:
    """Attaches a DepthReadAdapter to a block list via forward pre-hooks."""

    def __init__(self, blocks, adapter):
        assert adapter.ins < len(blocks)
        self.blocks, self.adapter = blocks, adapter
        self.handles = []
        self.stream = []

    @staticmethod
    def _get_h(args, kwargs):
        return args[0] if args else kwargs["hidden_states"]

    def _record(self, module, args, kwargs):
        if module is self.blocks[0]:
            self.stream = []
        self.stream.append(self._get_h(args, kwargs))
        return None

    def _inject(self, module, args, kwargs):
        self.stream.append(self._get_h(args, kwargs))
        h = self.stream[-1]
        mem = self.adapter.build_mem(self.stream)
        h_new = h + self.adapter.gamma * self.adapter.read(h, mem)
        self.stream = []
        if args:
            return ((h_new,) + tuple(args[1:]), kwargs)
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = h_new
        return (args, kwargs)

    def attach(self):
        assert not self.handles
        ins = self.adapter.ins
        for i in range(ins):
            self.handles.append(
                self.blocks[i].register_forward_pre_hook(self._record, with_kwargs=True))
        self.handles.append(
            self.blocks[ins].register_forward_pre_hook(self._inject, with_kwargs=True))

    def detach(self):
        for hd in self.handles:
            hd.remove()
        self.handles = []
        self.stream = []


MODEL_TABLE = {
    "gpt2": dict(layers=12, d=768, blocks="transformer.h",
                 lora_targets=("attn.c_proj", "mlp.c_proj")),
    "EleutherAI/pythia-410m": dict(layers=24, d=1024, blocks="gpt_neox.layers",
                                   lora_targets=("attention.dense", "mlp.dense_4h_to_h")),
    "EleutherAI/pythia-1.4b": dict(layers=24, d=2048, blocks="gpt_neox.layers",
                                   lora_targets=("attention.dense", "mlp.dense_4h_to_h")),
    "Qwen/Qwen2.5-0.5B": dict(layers=24, d=896, blocks="model.layers",
                              lora_targets=("self_attn.o_proj", "mlp.down_proj")),
    "Qwen/Qwen2.5-1.5B": dict(layers=28, d=1536, blocks="model.layers",
                              lora_targets=("self_attn.o_proj", "mlp.down_proj")),
    "gpt2-medium": dict(layers=24, d=1024, blocks="transformer.h",
                        lora_targets=("attn.c_proj", "mlp.c_proj")),
}


def get_blocks(model, model_id):
    mod = model
    for part in MODEL_TABLE[model_id]["blocks"].split("."):
        mod = getattr(mod, part)
    return list(mod)
