import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN
from config import TOP_K
from config import DEVICE



model = SimpleCNN().to(DEVICE)
weights = torch.load('weights/simple_cnn.pt', map_location=DEVICE, weights_only=True)
model.load_state_dict(weights)
model.eval()


def predict(image_tensor):


    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_classes = torch.topk(probs, k=TOP_K, dim=1)


    return {
        cls.item(): prob.item()
        for cls, prob in zip(top_classes[0], top_probs[0])
    }









if __name__ == "__main__":


    img = Image.open("image_mnist.png")

    convert_tensor = transforms.ToTensor()

    tensor_img = convert_tensor(img)


    print(f"Result for predict function on tensor of image : {predict(tensor_img)}")



