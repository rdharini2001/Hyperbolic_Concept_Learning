"""Check the installation and the GPU:  python scripts/check_env.py

Reports Python/torch/CUDA versions, the GPU and its compute capability, runs a real
bf16 kernel on it (this is what fails with "no kernel image is available" when a
Blackwell GPU meets a PyTorch build without sm_120 support), and checks that
CT-CLIP, the HF CLI and the weights are in place.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

ok = True


def report(label: str, good: bool, detail: str = "") -> None:
    global ok
    ok &= good
    print(f"  [{'ok' if good else '!!'}] {label}{(': ' + detail) if detail else ''}")


print(f"python {sys.version.split()[0]}")
import torch  # noqa: E402

print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cap = f"sm_{p.major}{p.minor}"
        report(f"GPU {i}", True, f"{p.name}, {p.total_memory / 2**30:.0f} GB, {cap}")
        built = tuple(map(int, torch.version.cuda.split(".")[:2])) if torch.version.cuda else (0, 0)
        if p.major >= 12 and built < (12, 8):
            report("Blackwell needs a CUDA >= 12.8 PyTorch build", False, "reinstall with the cu128 index")
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        report("bf16 matmul on GPU", bool(torch.isfinite((a @ a).float()).all()))
    except RuntimeError as e:
        report("bf16 matmul on GPU", False, str(e).splitlines()[0])
else:
    report("CUDA GPU", False, "none visible to torch (extraction would run on CPU: very slow)")

for mod in ("hypertrace", "nibabel", "transformers", "ct_clip", "transformer_maskgit", "vector_quantize_pytorch",
            "monai"):
    try:
        m = importlib.import_module(mod)
        report(f"import {mod}", True, getattr(m, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        report(f"import {mod}", False, f"{type(e).__name__}: {e}")
report("hf CLI", shutil.which("hf") is not None)

ckpt, bert = os.environ.get("HT_CTCLIP_CKPT"), os.environ.get("HT_CXR_BERT")
if ckpt:
    report("CT-CLIP checkpoint", Path(ckpt).is_file(), ckpt)
if bert:
    report("CXR-BERT folder", Path(bert, "config.json").is_file(), bert)
fm = os.environ.get("HT_FMCIB_CKPT")
if fm:
    report("FMCIB weights", Path(fm).is_file(), fm)
print("all good" if ok else "some checks failed (see !! above)")
sys.exit(0 if ok else 1)
