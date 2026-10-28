import os
import pytest 
import torch
from predictor import DigitPredictor
from config import TOP_K

@pytest.fixture
def unweighted_predictor():
    return DigitPredictor()


@pytest.fixture
def dummy_tensor():
    return torch.randn(1, 28, 28)


def test_predict_structure_and_types(unweighted_predictor, dummy_tensor):

    result = unweighted_predictor.predict(dummy_tensor)

    assert isinstance(result, dict), "Must return a dict"
    assert len(result) == TOP_K, f"Result must contains {TOP_K} classes"


    for digit, prob in result.items():
        assert isinstance(digit, int), f"The class {digit} must be an integer"
        assert isinstance(prob, float), f"The probability {prob} must be a float"

        assert 0.0 <= prob <= 1.0, "A probability must be between 0 and 1"


def test_predict_softmax_properties(unweighted_predictor, dummy_tensor):

    result = unweighted_predictor.predict(dummy_tensor)

    probabilities = list(result.values())

    assert sorted(probabilities, reverse=True) == probabilities, "The probabilities must be sorted"



def test_predict_with_real_weights(dummy_tensor):

    weights_path = "weights/simple_cnn.pt"

    assert os.path.exists(weights_path), f"The weights file {weights_path} is not found"

    predictor = DigitPredictor(weights_path)
    result = predictor.predict(dummy_tensor)

    assert len(result) == TOP_K
    assert not any(torch.isnan(torch.tensor(list(result.values())))), "The prediction with the real weights contains NaN"


