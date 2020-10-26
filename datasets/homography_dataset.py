from PIL import Image
from torch.utils.data.dataset import Dataset
import h5py


# https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
class HomographyDataset(Dataset):
    def __init__(self, h5_loc, group, transforms=None):
        self.h5_loc = h5_loc
        self.group = group
        self.dataset = None
        self.transforms = transforms

        with h5py.File(self.h5_loc, 'r') as file:
            self.dataset_len = len(file[self.group]['img'])

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_loc, 'r')[self.group]
        img = Image.fromarray(self.dataset['img'][idx])
        img_crp = Image.fromarray(self.dataset['img_crp'][idx])
        wrp_crp = Image.fromarray(self.dataset['wrp_crp'][idx])
        uv = self.dataset['uv'][idx]
        duv = self.dataset['duv'][idx]
        H = self.dataset['H'][idx]

        if self.transforms:
            img = self.transforms(img)
            img_crp = self.transforms(img_crp)
            wrp_crp = self.transforms(wrp_crp)
        
        return {
            'img': img,
            'img_crp':img_crp, 
            'wrp_crp': wrp_crp, 
            'uv': uv,
            'duv': duv,
            'H': H
        }

    def __len__(self):
        return self.dataset_len
