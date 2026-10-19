"""Convert intfloat/multilingual-e5-small to a fixed-length Core ML fp16 package.

Usage: uv run --project e5-small-coreml python e5-small-coreml/convert.py [sequence_length] [fp16|fp32]   (default 256 fp16; output: e5small_<precision>_<length>.mlpackage)
fp32 is for Intel Macs, which have no Neural Engine and run Core ML on CPU or GPU.
Pinned versions: see pyproject.toml (coremltools 9.0 is only validated up to torch 2.7).
"""
import sys
import numpy as np
import torch
import coremltools as ct
from transformers import AutoModel, AutoTokenizer

S = int(sys.argv[1]) if len(sys.argv) > 1 else 256
PRECISION = sys.argv[2] if len(sys.argv) > 2 else "fp16"
assert PRECISION in ("fp16", "fp32"), PRECISION
NAME = "intfloat/multilingual-e5-small"

base = AutoModel.from_pretrained(NAME, attn_implementation="eager", torch_dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(NAME)
# The default mask value (-inf in fp16) produces NaN on CPU and GPU; a finite value keeps all compute units correct.
base.get_extended_attention_mask = lambda m, shape, dtype=None, device=None: (1.0 - m[:, None, None, :].to(torch.float32)) * -1e4


class Encoder(torch.nn.Module):
    """Token ids in, a normalized 384-d sentence vector out (mean pooling and L2 normalization are inside the model)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0), dim=-1)


encoder = Encoder(base).eval()
example = tok(["passage: hello world"], padding="max_length", max_length=S, truncation=True, return_tensors="pt")
with torch.no_grad():
    traced = torch.jit.trace(encoder, (example["input_ids"], example["attention_mask"]))
model = ct.convert(
    traced,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS15,
    compute_precision=ct.precision.FLOAT16 if PRECISION == "fp16" else ct.precision.FLOAT32,
    inputs=[
        ct.TensorType("input_ids", shape=(1, S), dtype=np.int32),
        ct.TensorType("attention_mask", shape=(1, S), dtype=np.int32),
    ],
    outputs=[ct.TensorType("embedding")],
)
model.save(f"e5small_{PRECISION}_{S}.mlpackage")
