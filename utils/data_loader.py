
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PhysicsDataset(Dataset):
    def __init__(self, data_path, normalize=True):
        self.data = np.load(data_path)
        self.x = torch.tensor(self.data['x'], dtype=torch.float32)
        self.y = torch.tensor(self.data['y'], dtype=torch.float32)
        self.t = torch.tensor(self.data['t'], dtype=torch.float32)
        self.u = torch.tensor(self.data['u'], dtype=torch.float32)
        self.v = torch.tensor(self.data['v'], dtype=torch.float32)
        self.T = torch.tensor(self.data['T'], dtype=torch.float32)

        if normalize:
            self.normalize()

    def normalize(self):
        self.x = (self.x - self.x.mean()) / self.x.std()
        self.y = (self.y - self.y.mean()) / self.y.std()
        self.t = (self.t - self.t.mean()) / self.t.std()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t[idx], self.u[idx], self.v[idx], self.T[idx]

def create_dataloader(data_path, batch_size=32, shuffle=True):
    dataset = PhysicsDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
