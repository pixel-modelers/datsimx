import h5py
from torch.utils.data import Dataset
import numpy as np
import torch
#import math


class NlattsData(Dataset):

    def __init__(self, h5name, n_ex=5000):
        self.h5 = h5py.File(h5name, 'r')
        self.n_im = self.h5['masks'].shape[0]
        self.n_ex = n_ex
        self.xdim = self.h5['masks'].shape[-1]

        # n choose 2:
        #max_examples = math.comb(self.n_im, 2)

    def __len__(self):
        return self.n_ex

    def __getitem__(self, idx):
        assert idx < len(self)  # doesnt really matter .. 
        nlat = np.random.randint(1,5)

        #@if nlat == 0:
        #@    masks = [np.zeros((self.xdim, self.xdim))]
        #@else:
        inds = np.random.choice(self.n_im, size=nlat, replace=False)
        masks = [self.h5["masks"][ind] for ind in inds]

        combined = np.any(masks, axis=0).astype(np.float32)
        combined = torch.tensor(combined)[None]

        lab = 0. if nlat == 1 else 1.
        label = torch.tensor(lab)[None]
        return combined, label

class MlatData(Dataset):
    def __init__(self, h5name, n_ex=5000, mlat_chance=0.5):
        self.h5 = h5py.File(h5name, 'r')
        self.n_im = self.h5['masks'].shape[0]
        self.n_ex = n_ex
        self.mlat_chance = mlat_chance

        # n choose 2:
        #max_examples = math.comb(self.n_im, 2)

    def __len__(self):
        return self.n_ex

    def __getitem__(self, idx):
        assert idx < len(self)  # doesnt really matter .. 
        nlat = 2

        if np.random.random() < self.mlat_chance:
            inds = np.random.choice(self.n_im, size=nlat, replace=False)
            masks = [self.h5["masks"][ind] for ind in inds]
        else:
            ind1 = np.random.choice(self.n_im)
            mask1 = self.h5['masks'][ind1]
            mask2 = np.zeros_like(mask1)
            masks = mask1, mask2 
            masks = np.random.permutation(masks)

        combined = np.any(masks, axis=0).astype(np.float32)
        combined = torch.tensor(combined)[None]

        label = np.array(masks).astype(np.float32)
        label = torch.tensor(label)
        return combined, label


if __name__=="__main__":
    import sys
    fname =sys.argv[1]
    mlat = NlattsData(fname) 
    from IPython import embed;embed()

    mlat = MlatData(fname) 
    a,b = mlat[0]
    assert torch.all(a[0] == b[0] + b[1])

    print("Good")
