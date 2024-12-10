"""
Generate pseudo laue stills, saves as nexus master file
Unless --noWaveImg flag is passed, an HDF5 file will be written containing the per-pixel wavelengths
Need to get 3 files first:
wget https://raw.githubusercontent.com/dermen/e080_laue/master/air.stol
wget https://raw.githubusercontent.com/dermen/e080_laue/master/from_vukica.lam
iotbx.fetch_pdb 7lvc
"""


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--mosSpread", type=float, default=0.025, help="mosaic ang. spread in degrees (default=0.025)")
    parser.add_argument("--mosDoms", type=int, default=150, help="number of mosaic domains (default=150)")
    parser.add_argument("--div", type=float, default=0, help="divergence in mrad (default=0)")
    parser.add_argument("--divSteps", type=int, default=0, help="number of divergence steps in x,y (hence total num of steps prop. to square of this value)")
    parser.add_argument("--enSteps", type=int, default=322, help="Number of spectrum samples, use to upsample or downsample the specFile (default=322)")
    parser.add_argument("--testShot", action="store_true", help="only simulate a single shots")
    parser.add_argument("--ndev", type=int, default=1, help="number of GPU devices per compute node (default=1)")
    parser.add_argument("--run", type=int, default=1, help="run number in filename shot_R_#####.cbf")
    parser.add_argument("--mono", action='store_true', help="use the average wavelength to do mono simulation")
    parser.add_argument("--oversample", type=int, default=1, help="pixel oversample factor (increase if spots are sharp)")
    parser.add_argument("--numimg", type=int, default=180, help="number of images in 180 deg rotation")
    parser.add_argument("--noWaveImg", action="store_true", help="Dont write the wavelength-per-pixel image")
    parser.add_argument("--xtalSize", type=float, default=0.5, help="xtal size in mm")
    parser.add_argument("--gain", default=1, type=float, help="ADU per photon")
    parser.add_argument("--cuda", action="store_true", help="set DIFFBRAGG_USE_CUDA=1 env var to run CUDA kernel")
    parser.add_argument("--rotate", action="store_true", help="rotate the crystal between exposures")
    parser.add_argument("--numPhiSteps", type=int, default=10, help="number of mini-simulations to do between phi and phi+delta_phi if args.rotate is True")
    parser.add_argument("--totalDeg", type=float, help="total amount of crystal roataion in degrees (default=180)", default=180)
    parser.add_argument("--cbf", action="store_true", help="In addition to nexus, save a CBF file for each image simulated")
    parser.add_argument("--randomizerot", action="store_true",help="randomize the rotation of each shot, in which case totalDeg is irrelevant")
    args = parser.parse_args()

    import os
    import time
    import sys
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation
    
    from libtbx.mpi4py import MPI
    COMM = MPI.COMM_WORLD
    from simtbx.nanoBragg import utils
    from simtbx.diffBragg import utils as db_utils
    from simtbx.nanoBragg import nanoBragg
    from simtbx.modeling.forward_models import diffBragg_forward
    from dxtbx.model import DetectorFactory, BeamFactory, Crystal
    from scitbx.matrix import col, sqr
    from scitbx.array_family import flex
    from dxtbx.model import Experiment, ExperimentList


    from datsimx import make_nexus

    if args.cuda:
        os.environ["DIFFBRAGG_USE_CUDA"] = "1"

    total_rot = np.pi* args.totalDeg/180

    # convenience files from this repository
    this_dir = os.path.dirname(__file__)
    spec_file = os.path.join(this_dir, 'from_vukica.lam')  # intensity vs wavelength
    pdb_file = os.path.join(this_dir, '7lvc.pdb')
    air_name = os.path.join(this_dir, 'air.stol')
    total_flux=5e9
    beam_size_mm=0.01

    if COMM.rank==0:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        cmdfile = args.outdir+"/commandline_run%d.txt" % args.run
        with open(cmdfile, "w") as o:
            cmd = " ".join(sys.argv)
            o.write(cmd+"\n")
    COMM.barrier()

    # Rayonix model
    DETECTOR = DetectorFactory.simple(
        sensor='PAD',
        distance=200,  # mm
        beam_centre=(170, 170),  # mm
        fast_direction='+x',
        slow_direction='-y',
        pixel_size=(.08854, .08854),  # mm
        image_size=(3840, 3840))

    try:
        weights, energies = db_utils.load_spectra_file(spec_file)
    except:
        weights, energies = db_utils.load_spectra_file(spec_file, delim=" ")

    if args.enSteps is not None:
        from scipy.interpolate import interp1d
        wts_I = interp1d(energies, weights)# bounds_error=False, fill_value=0)
        energies = np.linspace(energies.min()+1e-6, energies.max()-1e-6, args.enSteps)
        weights = wts_I(energies)

    ave_en = np.mean(energies)
    ave_wave = utils.ENERGY_CONV / ave_en

    BEAM = BeamFactory.simple(ave_wave)

    Fcalc = db_utils.get_complex_fcalc_from_pdb(pdb_file, wavelength=ave_wave) #, k_sol=-0.8, b_sol=120) #, k_sol=0.8, b_sol=100)
    Famp = Fcalc.as_amplitude_array()

    water_bkgrnd = utils.sim_background(
        DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
        Fbg_vs_stol=None, sample_thick_mm=2.5, density_gcm3=1, molecular_weight=18)

    air_Fbg, air_stol = np.loadtxt(air_name).T
    air_stol = flex.vec2_double(list(zip(air_Fbg, air_stol)))
    air = utils.sim_background(DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
                            molecular_weight=14,
                            sample_thick_mm=5,
                            Fbg_vs_stol=air_stol, density_gcm3=1.2e-3)

    fdim, sdim = DETECTOR[0].get_image_size()
    img_sh = sdim, fdim
    water_bkgrnd = water_bkgrnd.as_numpy_array().reshape(img_sh)
    air = air.as_numpy_array().reshape(img_sh)

    if args.mono:
        energies = np.array([ave_en])
        weights = np.array([1])

    num_en = len(energies)
    fluxes = weights / weights.sum() * total_flux * len(weights)
    print("Simulating with %d energies" % num_en)
    print("Mean energy:", ave_wave)
    sg = Famp.space_group()
    print("unit cell, space group:\n", Famp, "\n")

    ucell = Famp.unit_cell()
    Breal = ucell.orthogonalization_matrix()
    # real space vectors
    a = Breal[0], Breal[3], Breal[6]
    b = Breal[1], Breal[4], Breal[7]
    c = Breal[2], Breal[5], Breal[8]
    CRYSTAL = Crystal(a, b, c, sg)

    randU = None
    if COMM.rank==0:
        randU = Rotation.random(random_state=0)
        randU = randU.as_matrix()
    randU = COMM.bcast(randU)
    CRYSTAL.set_U(randU.ravel())

    delta_phi = total_rot/ args.numimg

    gonio_axis = col((1,0,0))
    U0 = sqr(CRYSTAL.get_U())  # starting Umat

    Nabc = 100,100,100

    mos_spread = args.mosSpread
    num_mos = args.mosDoms
    device_Id = COMM.rank % args.ndev
    tsims = []

    saved_h5s = []
    
    random_states = None
    if args.randomizerot:
        if COMM.rank==0:
            np.random.seed()
            random_states = np.random.permutation(args.numimg*100)[:args.numimg]
        random_states = COMM.bcast(random_states)

    for i_shot in range(args.numimg):

        if i_shot % COMM.size != COMM.rank:
            continue
        tsim = time.time()
        print("Doing shot %d/%d" % (i_shot+1, args.numimg))
        Rphi = gonio_axis.axis_and_angle_as_r3_rotation_matrix(delta_phi*i_shot, deg=False)
        Uphi = Rphi * U0
        if args.randomizerot:
            Uphi = sqr(Rotation.random(1, random_states[i_shot]).as_matrix().ravel())
        CRYSTAL.set_U(Uphi)


        t = time.time()

        printout_pix=None
        from simtbx.diffBragg.device import DeviceWrapper
        with DeviceWrapper(device_Id) as _:


            out = diffBragg_forward(
                CRYSTAL, DETECTOR, BEAM, Famp, energies, fluxes,
                oversample=args.oversample, Ncells_abc=Nabc,
                mos_dom=num_mos, mos_spread=mos_spread, beamsize_mm=beam_size_mm,
                device_Id=device_Id,
                show_params=False, crystal_size_mm=args.xtalSize, printout_pix=printout_pix,
                verbose=COMM.rank==0, default_F=0, interpolate=0,
                mosaicity_random_seeds=None, div_mrad=args.div,
                divsteps=args.divSteps,
                show_timings=COMM.rank==0,
                nopolar=False, diffuse_params=None,
                delta_phi=delta_phi*180/np.pi if args.rotate else None,
                num_phi_steps = args.numPhiSteps if args.rotate else 1,
                perpixel_wavelen=not args.noWaveImg)

        if args.noWaveImg:
            img = out
            wave_img = h_img = k_img = l_img = None
        else:
            img, wave_img, h_img, k_img, l_img = out

        t = time.time()-t
        print("Took %.4f sec to sim" % t)
        if len(img.shape)==3:
            img = img[0]
            if wave_img is not None:
                wave_img = wave_img[0]
                h_img = h_img[0]
                k_img = k_img[0]
                l_img = l_img[0]

        img_with_bg = img +water_bkgrnd + air

        SIM = nanoBragg(detector=DETECTOR, beam=BEAM)
        SIM.beamsize_mm = beam_size_mm
        SIM.exposure_s = 1
        SIM.flux = total_flux
        SIM.adc_offset_adu = 10
        SIM.detector_psf_kernel_radius_pixels = 5
        SIM.detector_calibration_noice_pct = 3
        SIM.detector_psf_fwhm_mm = 0.1
        SIM.quantum_gain = args.gain
        SIM.readout_noise_adu = 3
        SIM.raw_pixels += flex.double((img_with_bg).ravel())
        SIM.add_noise()
        if args.cbf:
            cbf_name = os.path.join(args.outdir, "shot_%d_%05d.cbf" % (args.run, i_shot+1))
            SIM.to_cbf(cbf_name, cbf_int=True)
        
        noise_image = SIM.raw_pixels.as_numpy_array()
        
        h5_name = os.path.join(args.outdir, "shot_%d_%05d.h5" % (args.run, i_shot+1))
        #SIM.to_cbf(cbf_name, cbf_int=True)
        #img = SIM.raw_pixels.as_numpy_array().reshape(img_sh)
        SIM.free_all()
        del SIM
        #h5_name = cbf_name.replace(".cbf", ".h5")
        h = h5py.File(h5_name, "w")
        if wave_img is not None:
            h.create_dataset("wave_data", data=wave_img, dtype=np.float32, compression="lzf")
            h.create_dataset("h_data", data=h_img, dtype=np.float32, compression="lzf")
            h.create_dataset("k_data", data=k_img, dtype=np.float32, compression="lzf")
            h.create_dataset("l_data", data=l_img, dtype=np.float32, compression="lzf")
        h.create_dataset("noise_image", data=noise_image.astype(np.float32))
        h.create_dataset("delta_phi", data=delta_phi)
        h.create_dataset("Umat", data=CRYSTAL.get_U())
        h.create_dataset("Bmat", data=CRYSTAL.get_B())
        h.create_dataset("mos_spread", data=mos_spread)
        #h.create_dataset("img_with_bg", data=img_with_bg)
        h.close()
        saved_h5s.append(h5_name)
        tsim = time.time()-tsim
        if COMM.rank==0:
            print("TSIM=%f" % tsim)
        tsims.append(tsim)
        if args.testShot:
            break

    tsims = COMM.reduce(tsims)
    saved_h5s = COMM.reduce(saved_h5s)
    if COMM.rank==0:
        ave_tsim = np.median(tsims)
        print("Done", flush=True)
        print("Ave time to sim a shot=%f sec" % ave_tsim)
        El = ExperimentList()
        E = Experiment()
        E.detector = DETECTOR
        E.beam = BEAM
        El.append(E)
        geom_file = os.path.join(args.outdir, f"geom_run{args.run}.expt")
        El.as_file(geom_file)
        make_nexus.make_nexus(args.outdir, args.run, total_deg=args.totalDeg)
        

if __name__=="__main__":
    main()
