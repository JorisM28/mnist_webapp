import pytest 
from model import SimpleCNN
import torch


@pytest.fixture
def model():
    return SimpleCNN()


def test_model_output_shape(model):

    dummy_input = torch.randn(1, 1, 28, 28)

    output = model(dummy_input)

    assert output.shape == (1, 10)


def test_model_no_values(model):

    dummy_input = torch.randn(1, 1, 28, 28)

    output = model(dummy_input)

    assert not torch.isnan(output).any()
