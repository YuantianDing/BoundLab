import torch

import boundlab


def test_version():
    assert boundlab.__version__ == "0.1.0"


def test_torch_available():
    assert torch.__version__
