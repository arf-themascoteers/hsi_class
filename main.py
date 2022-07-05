from train import train
from test import test
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train(device)
test(device)