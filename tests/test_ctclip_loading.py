"""Checkpoint loading is robust to library-version differences but never silently wrong."""

import pytest
import torch
from torch import nn

from hypertrace.encoders.ctclip import load_ctclip_state


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 2)


def _save(tmp_path, sd):
    torch.save(sd, tmp_path / "ckpt.pt")
    return tmp_path / "ckpt.pt"


def test_loads_plain_and_wrapped_state_dicts(tmp_path):
    src = Toy()
    wrappers = (lambda s: s, lambda s: {"model": s},
                lambda s: {"state_dict": {f"module.{k}": v for k, v in s.items()}})
    for wrap in wrappers:
        dst = Toy()
        load_ctclip_state(dst, _save(tmp_path, wrap(src.state_dict())))
        assert torch.equal(dst.a.weight, src.a.weight)


def test_ignores_stale_bert_position_ids(tmp_path):
    sd = Toy().state_dict()
    sd["text_transformer.embeddings.position_ids"] = torch.arange(512)[None]  # saved by old transformers
    load_ctclip_state(Toy(), _save(tmp_path, sd))


def test_rejects_wrong_checkpoint(tmp_path):
    sd = Toy().state_dict()
    sd.pop("a.bias")
    sd["other.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="does not match"):
        load_ctclip_state(Toy(), _save(tmp_path, sd))
