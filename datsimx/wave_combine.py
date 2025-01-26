
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
from scipy.signal import savgol_filter, argrelmax
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
    strong_dset = h5_out.create_dataset("strong_masks", dtype=bool,**dset_args )

    loaders = {}
    h5_paths = {}
    meta_data =[]
    more_meta_data = []
    Umats = []
    Bmats = []
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

        raw_imgs = []
        peakwave_masks = []
        strong_masks = []
        for i in set(R['id']):
            Ri = R.select(R['id']==i)
            mask = utils.strong_spot_mask(Ri, Det)[0]
            E = El[i]
            path = E.imageset.paths()[0]
            img_idx = E.imageset.indices()[0]
            with h5py.File(path, 'r') as hmaster:
                data_ds = hmaster["entry/data/data"]
                data_src = data_ds.virtual_sources()[img_idx]
                with h5py.File(data_src.file_name, 'r') as hsingle:
                    raw_wave_img = hsingle['wave_data'][()]
                    energies = hsingle["energies"][()]
                    wavelens = utils.ENERGY_CONV / energies
                    fluxes = hsingle["fluxes"][()]
                    ave_en = np.average(energies, weights=fluxes)
                    var_en = np.average((energies-ave_en)**2, weights=fluxes)
                    sig_en = np.sqrt(var_en)

                    ave_wave = utils.ENERGY_CONV / ave_en
                    sig_wave = utils.ENERGY_CONV / sig_en
                    MASK = hsingle["MASK"][()]
                    center = hsingle["center"][()]
                    pixsize = hsingle["pixsize"][()]
                    distance = hsingle["distance"][()]
                    sgnum = hsingle["sgnum"][()]
                    ucpar = hsingle["ucpar"][()]
                    ucvol = hsingle["ucvol"][()]
                    Umat = hsingle["Umat"][()]
                    Bmat = hsingle["Bmat"][()]
            smooth_fluxes = savgol_filter(fluxes, window_length=15, polyorder=3)
            peak_flux_idx = argrelmax(smooth_fluxes, order=30)[0][0]
            peak_wave = wavelens[peak_flux_idx]
            
            temp = [peak_wave, ave_wave, sig_wave, center[0], center[1], pixsize, distance]
            meta_data.append(temp)
            more_meta_data.append( temp + list(ucpar) + [ucvol, sgnum] )
            Umats.append(list(Umat))
            Bmats.append(list(Bmat))
            # keep 1 percent bandwidth of spots off the peak:
            wave_min, wave_max = peak_wave-peak_wave*0.005, peak_wave + peak_wave*0.005
            is_peak_wave = (~np.isnan(raw_wave_img)) * \
                    (raw_wave_img >= wave_min) * (raw_wave_img <= wave_max) * mask
            #a2,na2 =label(mask2)
            #for sY, sX in find_objects(a):
            #    sub_waveimg = raw_wave_img[sY, sX]
            #    sub_mask = mask[sY, sX]
            #    wave_mask = sub_mask * ~np.isnan(sub_waveimg)
            #    wave_vals = sub_waveimg[wave_mask]
            #    ave_wave = np.mean(wave_vals)
            #    #if np.any((wave_min <= wave_vals) * (wave_max <= wave_vals)):
            #    if wave_min <= ave_wave <= wave_max:
            #        mask3[sY,sX] = sub_mask
            assert np.any(is_peak_wave)

            if path not in loaders:
                loader = dxtbx.load(path)
                loaders[path] = loader
            raw_img = loaders[path].get_raw_data(img_idx)[0].as_numpy_array()
            raw_img *= MASK
            ds_img = counter_utils.process_image(raw_img, max_pool, useSqrt=True)[0]
            ds_mask = counter_utils.process_image(is_peak_wave, max_pool, useSqrt=False)[0]
            ds_mask = ds_mask.round().numpy().astype(bool)
            
            strong_mask = counter_utils.process_image(mask, max_pool, useSqrt=False)[0]
            strong_mask = strong_mask.round().numpy().astype(bool)
            
            print(f"file {ref_f} ({i_ref_f+1}/{len(refl_files)}); loaded shot {i+1}/{len(El)}")
            peakwave_masks.append(ds_mask)
            strong_masks.append(strong_mask)
            raw_imgs.append(ds_img)
        counts += len(raw_imgs)
        
        start = 0 if start is None else imgs_dset.shape[0]
        imgs_dset.resize((counts, args.cropdim, args.cropdim))
        imgs_dset[start: start+len(raw_imgs)] = np.array(raw_imgs)

        masks_dset.resize((counts, args.cropdim, args.cropdim))
        masks_dset[start: start+len(peakwave_masks)] = np.array(peakwave_masks).astype(bool)
        
        strong_dset.resize((counts, args.cropdim, args.cropdim))
        strong_dset[start: start+len(strong_masks)] = np.array(strong_masks).astype(bool)

rank_files = COMM.gather(rank_file)
rank_counts = COMM.gather(counts)

# the reduce order should match the gather order
meta_data = COMM.reduce(meta_data)
more_meta_data = COMM.reduce(more_meta_data)
Umats = COMM.reduce(Umats)
Bmats = COMM.reduce(Bmats)

if COMM.rank==0:
    print(f"Aggregating results into single file {args.h5out}... ")
    total_counts = sum(rank_counts)
    imgs_layout = h5py.VirtualLayout(shape=(total_counts,args.cropdim, args.cropdim), dtype=np.float32)
    masks_layout = h5py.VirtualLayout(shape=(total_counts,args.cropdim, args.cropdim), dtype=bool)
    strong_layout = h5py.VirtualLayout(shape=(total_counts,args.cropdim, args.cropdim), dtype=bool)
    start = 0
    import os
    for rf,nimg in zip(rank_files, rank_counts):
        #rf = os.path.abspath(rf)
        if nimg ==0:
            continue
        for key,layout in zip(["raw_images", "masks","strong_masks"],[imgs_layout, masks_layout, strong_layout]):
            vsource = h5py.VirtualSource(rf, key, shape=(nimg,args.cropdim, args.cropdim))
            layout[start:start+nimg] = vsource
        start += nimg
    with h5py.File(args.h5out, "w") as h5_out:
        h5_out.create_virtual_dataset("raw_images", imgs_layout )
        h5_out.create_virtual_dataset("peakwave_masks", masks_layout )
        h5_out.create_virtual_dataset("strong_masks", strong_layout )
        meta_ds = h5_out.create_dataset("meta_data", data=meta_data)
        more_meta_ds = h5_out.create_dataset("more_meta_data", data=more_meta_data)
        h5_out.create_dataset("Umats", data=Umats)
        h5_out.create_dataset("Bmats", data=Bmats)
    print("Done.")

