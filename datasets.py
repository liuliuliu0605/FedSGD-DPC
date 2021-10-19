import torch
from torch.utils.data import Dataset
import os


class train_dataset(Dataset):
    def __init__(self, root, user_name, percent=1.0):
        super(train_dataset, self).__init__()
        data, targets = torch.load(os.path.join(root, "%d.pt" % user_name))
        size = int(len(data)*percent)
        step = int(len(data) / size)
        self.data, self.targets = data[::step], targets[::step]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return torch.Tensor.float(img), target

    def __len__(self):
        return len(self.targets)


class test_dataset(Dataset):
    def __init__(self, root, percent=1.0):
        super(test_dataset, self).__init__()
        data, targets = torch.load(root)
        size = int(len(data) * percent)
        step = int(len(data) / size)
        self.data, self.targets = data[::step], targets[::step]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = torch.Tensor.float(img)
        return img, target

    def __len__(self):
        return len(self.targets)


