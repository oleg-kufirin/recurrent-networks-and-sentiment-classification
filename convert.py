# convert.py
import torch
from part3 import Network, PreProcessing
device = torch.device("cpu" if torch.cuda.is_available() else "cuda:0")
net = Network().to(device)
print(f'Loading model..')
net.load_state_dict(torch.load('model.pth', map_location=torch.device(device)))
print(f'Saving model as {device}.')
torch.save(net.state_dict(), './model.pth')