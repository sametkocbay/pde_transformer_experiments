#%%
import torch
from pdetransformer.core.mixed_channels import PDETransformer
import matplotlib.pyplot as plt

# 1. Setup Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. Load Model
# We use the 'mc-s' (mixed-channels small) model as in your example
print("Loading PDE-Transformer Model...")
model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').to(device)
model.eval()

