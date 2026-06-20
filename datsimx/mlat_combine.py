
from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("glob_s",type=str,help="glob for stills  process output.strong_filenames (strong spot refls)")
ap.add_argument("h5out", type=str, help="hdf5 file for storing downsampled data")
ap.add_argument("--cropdim", type=int, default=832, help="dimension of downsampled images (default=832)")
args = ap.parse_args()
import glob
import sys

from dials.array_family import flex
from dxtbx.model import ExperimentList
import dxtbx
from simtbx.diffBragg import utils
import numpy as np
import h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD

from resonet.utils import counter_utils


def get_ds_fact(xdim):
    """given detector xdimension, return downsample factor"""
    if xdim == 2463:  # Pilatus 6M
        center_ds_fact = 3
    elif xdim == 3840:
        center_ds_fact = 4
    elif xdim == 4096:  # Mar
        center_ds_fact = 5
    else:  # Eiger
        center_ds_fact = 5
    return center_ds_fact
    
refl_files = glob.glob(args.glob_s)
if COMM.rank==0:
    print(f"found {len(refl_files)} files")
if not refl_files:
    exit()


counts = 0  
start = None
rank_file = args.h5out + f".rank{COMM.rank}"
dset_args = {"maxshape":(None, args.cropdim, args.cropdim), 
             "shape":(1,args.cropdim, args.cropdim), 
             "chunks": (1,args.cropdim, args.cropdim)}

with h5py.File(rank_file, "w") as h5_out:

    imgs_dset = h5_out.create_dataset("raw_images", dtype=np.float32, **dset_args)
    masks_dset = h5_out.create_dataset("masks", dtype=bool,**dset_args )

    for i_ref_f, ref_f in enumerate(refl_files):
        if i_ref_f % COMM.size != COMM.rank:
            continue
        R = flex.reflection_table.from_file(ref_f)
        exp_f = ref_f.replace(".refl", ".expt")
        El = ExperimentList.from_file(exp_f, False)
        Det = El.detectors()[0]
        xdim,ydim = Det[0].get_image_size()
        center_ds_fact = get_ds_fact(xdim)
        #cropdim = min(xdim, ydim) // center_ds_fact - 1
        if COMM.rank==0:
            print(f"Using downsamp factor {center_ds_fact} for {exp_f} images")
        max_pool = counter_utils.mx_gamma(stride=center_ds_fact, dim=args.cropdim)

        loaders = {}
        raw_imgs = []
        masks = []
        for i in set(R['id']):
            Ri = R.select(R['id']==i)
            mask = utils.strong_spot_mask(Ri, Det)[0]
            E = El[i]
            path = E.imageset.paths()[0]
            img_idx = E.imageset.indices()[0]
            if path not in loaders:
                loader = dxtbx.load(path)
                loaders[path] = loader
            raw_img = loaders[path].get_raw_data(img_idx)[0].as_numpy_array()
            ds_img = counter_utils.process_image(raw_img, max_pool, useSqrt=False)[0]
            ds_mask = counter_utils.process_image(mask, max_pool, useSqrt=False)[0]
            ds_mask = ds_mask.round().numpy().astype(bool)
            
            print(f"file {ref_f} ({i_ref_f+1}/{len(refl_files)}); loaded shot {i+1}/{len(El)}")
            masks.append(ds_mask)
            raw_imgs.append(ds_img)
        counts += len(raw_imgs)
        
        start = 0 if start is None else imgs_dset.shape[0]
        imgs_dset.resize((counts, args.cropdim, args.cropdim))
        imgs_dset[start: start+len(raw_imgs)] = np.array(raw_imgs)

        masks_dset.resize((counts, args.cropdim, args.cropdim))
        masks_dset[start: start+len(masks)] = np.array(masks).astype(bool)

rank_files = COMM.gather(rank_file)
rank_counts = COMM.gather(counts)

if COMM.rank==0:
    print(f"Aggregating results into single file {args.h5out}... ")
    total_counts = sum(rank_counts)
    imgs_layout = h5py.VirtualLayout(shape=(total_counts,args.cropdim, args.cropdim), dtype=np.float32)
    masks_layout = h5py.VirtualLayout(shape=(total_counts,args.cropdim, args.cropdim), dtype=bool)
    start = 0
    for rf,nimg in zip(rank_files, rank_counts):
        if nimg ==0:
            continue
        for key, layout in zip(["raw_images", "masks"],[imgs_layout, masks_layout]):
            vsource = h5py.VirtualSource(rf, key, shape=(nimg,args.cropdim, args.cropdim))
            layout[start:start+nimg] = vsource
        start += nimg
    with h5py.File(args.h5out, "w") as h5_out:
        h5_out.create_virtual_dataset("raw_images", imgs_layout )
        h5_out.create_virtual_dataset("masks", masks_layout )
    print("Done.")

