from kan import KAN

# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm



# Define model
model = KAN([28 * 28, 64, 10])
model.load_state_dict(torch.load('./cp/model_epoch_9.pt'))


inp = torch.ones(1, 28*28)

outfilename = './kan.onnx'
torch.onnx.export(
    model,
    (inp),
    outfilename,
    verbose=True,
    opset_version=15,
)
