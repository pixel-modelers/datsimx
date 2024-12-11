import h5py
from torch.utils.data import Dataset
import numpy as np
import torch
#import math



class WaveData(Dataset):

    def __init__(self, h5name, maximg=None):
        self.h5 = h5py.File(h5name, 'r')
        self.wave_masks = self.h5["peakwave_masks"]
        self.images = self.h5["raw_images"]
        if maximg is not None:
            assert maximg < self.wave_masks.shape[0]
        self.maximg = maximg

    def __len__(self):
        if self.maximg is not None:
            return self.maximg
        else:
            return self.wave_masks.shape[0]

    def __getitem__(self, idx):
        wave = self.wave_masks[idx]
        img = self.images[idx]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        data = torch.tensor(img[None])
        label = torch.tensor(wave[None]).float()
        return data, label



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
        self.ydim = self.h5['masks'].shape[-2]
        self.xdim = self.h5['masks'].shape[-1]

        self.n_ex = n_ex
        self.mlat_chance = mlat_chance

        # n choose 2:
        #max_examples = math.comb(self.n_im, 2)

    def __len__(self):
        return self.n_ex

    def __getitem__(self, idx):
        assert idx < len(self)  # doesnt really matter .. 
        nlat = 2
        inds = np.random.choice(self.n_im, size=nlat, replace=False)

        #if np.random.random() < self.mlat_chance:
        #    #nlat = np.random.choice([2,3])
        #    nlat = 2
        #else:
        #    nlat = 1
        #    inds = [np.random.choice(self.n_im)]
        #label = np.zeros((1,self.ydim,self.xdim), int)
        #combined = np.zeros(label.shape, bool)
        #for i_ind,ind in enumerate(inds):
        #    #mask = self.h5['masks'][ind]
        #    combined = np.logical_or(combined , mask)
        #    #label[mask[None]] = i_ind+1

        masks = [self.h5["masks"][ind] for ind in inds]
        #else:
        #    ind1 = np.random.choice(self.n_im)
        #    mask1 = self.h5['masks'][ind1]
        #    mask2 = np.zeros_like(mask1)
        #    masks = mask1, mask2 
        #    masks = np.random.permutation(masks)

        combined = np.any(masks, axis=0)[None] #.astype(np.float32)
        combined = torch.tensor(combined.astype(np.float32))

        label = np.array(masks).astype(np.float32)
        label = torch.tensor(label)
        #if nlat < 5:
        #    if nlat  in [3,4]:
        #        label = torch.cat( (label, label[:(5-nlat)]), axis=0)
        #    elif nlat == 1:
        #        label = torch.cat( (label, label, label, label, label), axis=0)
        #    else:
        #        label = torch.cat( (label, label, label[:1]), axis=0)
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
