from torchvision.datasets import EuroSAT
from torchvision import transforms
import os

# Set root directory relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")

print(f"Saving data to: {data_dir}")

# Download EuroSAT (RGB version)
dataset = EuroSAT(
    root=data_dir,
    download=True,
    transform=transforms.ToTensor()
)

print(f"Downloaded {len(dataset)} images to {os.path.join(data_dir, 'EuroSAT')}")

# Verify directory contents
euro_sat_dir = os.path.join(data_dir, "EuroSAT", "2750")
if os.path.exists(euro_sat_dir):
    classes = os.listdir(euro_sat_dir)
    print(f"Found classes: {classes}")
else:
    print(f"Warning: Expected EuroSAT directory {euro_sat_dir} not found!")
