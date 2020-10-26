import cv2
import pandas as pd
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, df_path: str, transforms=None):
        self.df = pd.read_pickle(df_path)
        self.transforms = transforms

    def __getitem__(self, idx):
        x_t = cv2.imread(self.df['filename'][idx])
        x_tp1

        if self.transforms:
            x_t = self.transforms(x_t)

        return x

    def __len__(self):
        return len(self.df)
