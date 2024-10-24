import glob
import sys

import h5py
import os
from dxtbx.model import ExperimentList
import numpy as np



def make_nexus(folder, run, total_deg=180):
    """
    folder is a sim_laue_basic.py output folder, 
    and run is an integer run number (args.run argument to sim_laue_basic)
    Finds all files shot_{run}_#####.h5 in {folder} and combines them into a master nexus file
    """
    glob_s = os.path.join(folder, f"shot_{run}_*.h5")

    fnames = glob.glob(glob_s)

    def get_num(f):
        return int(f.split("_")[-1].split(".")[0])
    fnames = sorted(fnames, key=get_num)
    delta_phi = total_deg/len(fnames)

    img_nums = [get_num(f) for f in fnames]

    phi = np.arange(len(fnames))*delta_phi 

    geom_file = os.path.join(folder, "geom.expt")
    El = ExperimentList.from_file(geom_file, False)
    Detector = El[0].detector
    xdim, ydim = Detector[0].get_image_size()
    pix_mm = Detector[0].get_pixel_size()[0]
    wavelen = El[0].beam.get_wavelength()
    dist_mm = Detector[0].get_distance()
    img_shape = ydim, xdim
    nimg = len(fnames)

    dt = h5py.special_dtype(vlen=str)
    master_file = os.path.join(folder, f"run_{run}_master.h5")

    h = h5py.File(master_file, 'w')
    data_layout = h5py.VirtualLayout(shape=(len(fnames),) + img_shape, dtype=np.float32)
    for i_f, f in enumerate(fnames):
        print(f"Processing file {i_f+1}/{len(fnames)}", end="\r", flush=True)
        data_source = h5py.VirtualSource(os.path.abspath(f), "noise_image", shape=(1,) + img_shape)
        data_layout[i_f:i_f+1] = data_source
    print("\n")


    entry = h.create_group("entry")
    entry.attrs["NX_class"] = "NXentry"
    entry.create_dataset('definition', data="NXmx", dtype=dt)
    inst = entry.create_group("instrument")
    inst.attrs["NX_class"] = "NXinstrument"
    det = inst.create_group("detector")
    det.attrs["NX_class"] = "NXdetector"
    detspec = det.create_group("detectorSpecific")
    detspec.attrs["NX_class"] = "NXcollection"
    samp = entry.create_group("sample")
    samp.attrs["NX_class"] = "NXsample"
    data = entry.create_group("data")
    data.attrs["NX_class"] = "NXdata"
    beam = inst.create_group("beam")
    beam.attrs["NX_class"] = "NXbeam"

    wavelen_dset = beam.create_dataset("incident_wavelength", data=wavelen)
    wavelen_dset.attrs["units"] = "angstrom"

    vdata = data.create_virtual_dataset("data", data_layout)
    vdata.attrs["image_nr_high"] = len(fnames)
    vdata.attrs["image_nr_low"] = 1

    # Goniometer and scan:
    ds = h.create_dataset("/entry/sample/phi", data=phi)
    ds.attrs["units"] = "deg"
    ds.attrs["transformation_type"] = "rotation"
    ds.attrs["vector"] = np.array([-1,0,0])
    samp.create_dataset('depends_on', data="phi".encode("utf-8"), dtype=dt)
    samp.create_dataset('name', data="goniometer".encode("utf-8"), dtype=dt)
    # Detector
    thick_dset = det.create_dataset("sensor_thickness", data=0.00032)
    thick_dset.attrs["units"] = "m"
    det.create_dataset("sensor_material", data="Si", dtype=dt)
    xdim_dset = detspec.create_dataset("x_pixels_in_detetector", data=xdim)
    ydim_dset = detspec.create_dataset("y_pixels_in_detetector", data=ydim)
    xdim_dset.attrs["units"] = "pixels"
    ydim_dset.attrs["units"] = "pixels"

    xpix_mm_dset = det.create_dataset("x_pixel_size", data=pix_mm/1000)
    ypix_mm_dset = det.create_dataset("y_pixel_size", data=pix_mm/1000)
    ypix_mm_dset.attrs["units"] = "m"
    xpix_mm_dset.attrs["units"] = "m"

    beam_x, beam_y = Detector[0].get_beam_centre_px(El[0].beam.get_s0())
    beamx_dset = det.create_dataset("beam_center_x", data=beam_x)
    beamy_dset = det.create_dataset("beam_center_y", data=beam_y)
    beamx_dset.attrs["units"] = "pixels"
    beamy_dset.attrs["units"] = "pixels"


    dist_dset = det.create_dataset("detector_distance", data=dist_mm/1000)
    dist_dset.attrs["units"] = "m"

    vec = beam_x*pix_mm, beam_y*pix_mm, dist_mm
    eiger = det.create_group("eiger")
    eiger.attrs["NX_class"] = "NXdetector_module"
    trans_path = '/entry/instrument/detector/eiger/trans'
    trans = eiger.create_dataset( "trans", data=[0.])
    trans.attrs["depends_on"]= "."
    trans.attrs["equipment"]= "detector"
    trans.attrs["equipment_component"]= "detector_level_0"
    trans.attrs["offset"] = np.array([0,0,0])
    trans.attrs["offset_units"] = "mm"
    trans.attrs["transformation_type"] = "rotation"
    trans.attrs["units"] = "deg"
    trans.attrs["vector"]= np.array([0,0,-1])
    mod_vals_x = trans_path , np.array(vec), 'mm', 'translation', 'mm',np.array([-1,0,0])
    mod_vals_y = trans_path, np.array(vec), 'mm', 'translation', 'mm',np.array([0,-1,0])
    mod_keys = ['depends_on','offset','offset_units','transformation_type','units','vector']
    fast_key = "fast_pixel_direction"
    slow_key = "slow_pixel_direction"
    fast = eiger.create_dataset(f"{fast_key}", data=[pix_mm])
    slow = eiger.create_dataset(f"{slow_key}", data=[pix_mm])
    for k,val_x, val_y in zip(mod_keys, mod_vals_x, mod_vals_y):
        fast.attrs[k] = val_x
        slow.attrs[k] = val_y
    eiger.create_dataset("data_origin", data=[0,0])
    eiger.create_dataset("data_size", data=[int(xdim), int(ydim)])
    h.close()
    print(f"Wrote file {master_file}.")


if __name__=="__main__":
    folder = sys.argv[1]
    run = int(sys.argv[2])
    make_nexus(folder, run)
