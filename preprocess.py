import torch
from torchvision import transforms
from PIL import Image, ImageOps


preprocessing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1307],
        std=[0.3081]
    )
])


def resize_and_center(img_cropped):

    size = img_cropped.size
    
    max_ = max(size)
    s = 20 / max_
    new_size = (round(s * size[0]), round(s * size[1]))
    X_offset = (28 - new_size[0]) // 2
    Y_offset = (28 - new_size[1]) // 2

    img_resized = img_cropped.resize(new_size, Image.Resampling.LANCZOS)
    
    final_img = Image.new("L", (28, 28), color=0)
    
    final_img.paste(img_resized, (X_offset, Y_offset))
    
    return final_img


def preprocess(img1):

    if img1.mode in ("RGBA", "LA") or (img1.mode == "P" and "transparency" in img1.info):

        background = Image.new("RGB", img1.size, (255, 255, 255))
        background.paste(img1, mask=img1.split()[-1]) 
        img1 = background

    img1_grey = img1.convert("L")
    img1_inv = ImageOps.invert(img1_grey)

    bbox = img1_inv.getbbox()

    if bbox is None:
        final_img = Image.new("L", (28, 28), color=0)
    else:
        new_img1 = img1_inv.crop(bbox)
        final_img = resize_and_center(new_img1)

    return final_img



def final_transform(img): 

    img_tensor = preprocessing(img)

    img_tensor_batch = img_tensor.unsqueeze(0)

    return img_tensor_batch