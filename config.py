import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MODEL_WEIGHTS_PATH = "weights/simple_cnn.pt"


NUM_CLASSES = 10

TOP_K = 3