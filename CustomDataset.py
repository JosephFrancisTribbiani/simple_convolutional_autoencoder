import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np

class FacesDataset(Dataset):
  def __init__(self, data, transform=None):
    self.data = data
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image = self.data[idx].transpose(2, 0, 1) / 255.
    image = torch.tensor(image, dtype=torch.float32)
    if self.transform:
      image = self.transform(image)
    return (image, ) 