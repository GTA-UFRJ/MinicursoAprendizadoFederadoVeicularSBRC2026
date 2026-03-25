from torch.utils.data  import Dataset

class SignDataset(Dataset):

    def __init__(self,
                 data,
                 labels,
                 transform=None):

        self.data = data
        self.targets =  labels
        self.classes = sorted(list(set(self.targets)))
        self.transform =  transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self,
                    idx):

        x = self.data[idx]
        y = self.targets[idx]

        if self.transform is not None:

            x = self.transform(x)

        return x, y
