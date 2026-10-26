from hypertrace.encoders.base import CTVisionLanguageModel, LesionEncoder


def build_encoder(name: str, **kwargs) -> CTVisionLanguageModel | LesionEncoder:
    """Factory so configs can pick an encoder without importing CT-CLIP / MONAI eagerly."""
    if name == "ctclip":
        from hypertrace.encoders.ctclip import FrozenCTCLIP

        return FrozenCTCLIP(**kwargs)
    if name == "fmcib":
        from hypertrace.encoders.fmcib import FrozenFMCIB

        return FrozenFMCIB(**kwargs)
    if name == "fake":
        from hypertrace.encoders.fake import FakeVLM

        return FakeVLM(**kwargs)
    if name == "fake_lesion":
        from hypertrace.encoders.fake import FakeLesionEncoder

        return FakeLesionEncoder(**kwargs)
    raise ValueError(f"Unknown encoder '{name}' (options: ctclip, fmcib, fake, fake_lesion)")


build_vlm = build_encoder  # backwards-compatible name

__all__ = ["CTVisionLanguageModel", "LesionEncoder", "build_encoder", "build_vlm"]
