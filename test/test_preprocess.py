import pytest
import torch
from PIL import Image
from preprocess import resize_and_center, preprocess, final_transform

@pytest.fixture
def dummy_rgba_image():

    img = Image.new("RGBA", (200, 200), (255, 255, 255, 0))

    for x in range(80, 120):
        for y in range(80, 120):
            img.putpixel((x, y), (0, 0, 0, 255))
    return img

@pytest.fixture
def blank_image():
    return Image.new("RGB", (100, 100), (255, 255, 255))





def test_preprocess_output_mode_and_size(dummy_rgba_image):

    processed = preprocess(dummy_rgba_image)

    assert isinstance(processed, Image.Image), "The result must be a PIL image"
    assert processed.mode == "L", f"The mode must be 'L', received {processed.mode}"
    assert processed.size == (28, 28), f"The size must be (28, 28), received {processed.size}"


def test_preprocess_blank_canvas(blank_image):

    processed = preprocess(blank_image)

    assert processed.size == (28, 28)

    extrema = processed.getextrema()
    assert extrema == (0, 0), "A blank canvas must product an entirely black image"

def test_resize_and_center_proportions():

    sample = Image.new("L", (100, 50), color=255)
    centered = resize_and_center(sample)

    assert centered.size == (28, 28)


def test_final_transform_tensor_contract():

    img_pil = Image.new("L", (28, 28), color=128)
    tensor = final_transform(img_pil)

    assert isinstance(tensor, torch.Tensor), "The result must be a torch.Tensor"

    assert tensor.shape == (1, 1, 28, 28), f"Expected form (1, 1, 28, 28), obtained {tensor.shape}"

    assert tensor.dtype == torch.float32, "The tensor must be of type float32"

    assert not torch.isnan(tensor).any(), "The tensor contains NaN values"