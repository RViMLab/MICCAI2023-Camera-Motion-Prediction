from torch.utils.data import Dataset
import cv2
from typing import Callable

class SquenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prefix: str, transforms: Callable=None):
        self.df = df
        self.prefix = prefix
        self.transforms = transforms

    def getitem(self, idx):
        file_seq = self.df['file_seq'][idx]
        img_seq = []

        for file in file_seq:
            img = cv2.imread(os.path.join(self.prefix, self.df['path'][idx], file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_seq.append(img)

        return img_seq

    def __len__(self):
        return len(self.df)
