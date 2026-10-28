import torch
from model import SimpleCNN
from config import TOP_K, DEVICE

class DigitPredictor:
    def __init__(self, model_path=None):


        self.model = SimpleCNN().to(DEVICE)

        if model_path:
            weights = torch.load(model_path, map_location=DEVICE, weights_only=True)
            self.model.load_state_dict(weights)


        self.model.eval()


    def predict(self, image_tensor):

        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_classes = torch.topk(probs, k=TOP_K, dim=1)


        return {
            cls.item() : prob.item()
            for cls, prob in zip(top_classes[0], top_probs[0])
        }

if __name__== "__main__":
    from PIL import Image
    from torchvision import transforms

    predictor = DigitPredictor("weights/simple_cnn.pt")

    img = Image.open("image_mnist.png")
    tensor_img = transforms.ToTensor()(img)
    
    print(f"Résultat : {predictor.predict(tensor_img)}")