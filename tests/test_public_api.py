import neutro
import neutro.layers as layers
import neutro.models as models
import neutro.optimizers as optimizers


def test_layer_exports_are_defined():
    missing = [name for name in layers.__all__ if not hasattr(layers, name)]
    assert missing == []


def test_model_exports_are_defined():
    missing = [name for name in models.__all__ if not hasattr(models, name)]
    assert missing == []


def test_optimizer_exports_are_defined():
    missing = [name for name in optimizers.__all__ if not hasattr(optimizers, name)]
    assert missing == []


def test_gpt_tiny_aliases():
    assert models.GPT1Tiny is models.GPT1
    assert models.GPT2Tiny is models.GPT2


def test_top_level_exports_are_defined():
    missing = [name for name in neutro.__all__ if not hasattr(neutro, name)]
    assert missing == []
