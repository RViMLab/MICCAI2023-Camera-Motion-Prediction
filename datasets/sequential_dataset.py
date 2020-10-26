from PIL import Image
from torch.utils.data.dataset import Dataset
import h5py


# https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
class SequentialDataset(Dataset):
    def __init__(self, h5_loc, group, transforms=None):
        self.h5_loc = h5_loc
        self.group = group
        self.dataset = None
        self.transforms = transforms

        with h5py.File(self.h5_loc, 'r') as file:
            self.dataset_len = len(file[self.group]['seq'])

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_loc, 'r')[self.group]
        seq = []

        uv = self.dataset['uv'][idx]

        if self.transforms:
            for i in range(len(self.dataset['seq'][idx])):
                seq.append(self.transforms(Image.fromarray(self.dataset['seq'][idx][i])))

        return {
            'seq': seq,
            'uv': uv
        }

    def __len__(self):
        return self.dataset_len
