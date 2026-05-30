import pytest
from neutro.losses import get

def test_get_vae_requires_instance():
    with pytest.raises(ValueError, match="VAELoss\\(model, \\.\\.\\.\\)"):
        get('vae')
