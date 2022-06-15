from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class BearDataset(Dataset):
    def __init__(self, file, transforms=None, normalize=True):
        self.file = file
        self.dataframe = pd.read_csv(self.file, header=None, skiprows=1)
        self.transforms = transforms
        self.normalize = normalize

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, i):
        path, label = self.dataframe[0][i], self.dataframe[1][i]
        img = imread(path)
        if self.normalize:
            x_mean = img.mean()
            x_max = img.max()
            img = (img - x_mean) / x_max
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label


def get_data_loader(batch_size, transforms, normalize, train=True, shuffle=True):

    if train:
        dataset = BearDataset('./data/train.csv', transforms, normalize)
    else:
        dataset = BearDataset('./data/test.csv', transforms, normalize)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print('got dataloader')
    return data_loader
