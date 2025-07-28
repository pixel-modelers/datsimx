
def main():

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="""Generate diffraction images and save them as Nexus master files.
Optionally, an HDF5 file containing per-pixel wavelengths will be written unless --noWaveImg is specified.

        """,
        epilog="""
Example usage:
python mx_simulate.py my_output_directory --numimg 360 --mosSpread 0.05 --cuda

For two-crystal domain simulation:
python mx_simulate.py my_output_directory_split --numimg 180 --splitPhiStart 45 --splitPhiEnd 135 --splitRotAxis 0 1 0 --splitRotAngle 0.2 --splitScale 0.75
        """
    )
    parser.add_argument(
        "outdir",
        type=str,
        help="Output directory for simulated images and Nexus master file.",
    )

    # Simulation Parameters Group
    sim_params_group = parser.add_argument_group("Simulation Parameters")
    sim_params_group.add_argument(
        "--mosSpread",
        type=float,
        default=0.1,
        help="Mosaic angular spread in degrees (FWHM). Default is 0.1 degrees.",
    )
    sim_params_group.add_argument(
        "--mosDoms",
        type=int,
        default=150,
        help="Number of mosaic domains to simulate. Default is 150.",
    )
    sim_params_group.add_argument(
        "--div",
        type=float,
        default=0,
        help="Beam divergence in mrad. Default is 0 (no divergence).",
    )
    sim_params_group.add_argument(
        "--divSteps",
        type=int,
        default=0,
        help="Number of divergence steps in x and y directions. Total number of steps is proportional to the square of this value. Default is 0.",
    )
    sim_params_group.add_argument(
        "--enSteps",
        type=int,
        default=100,
        help="Number of spectrum samples. Use to upsample or downsample the provided spectrum file. Default is 100.",
    )
    sim_params_group.add_argument(
        "--oversample",
        type=int,
        default=1,
        help="Pixel oversampling factor. Increase this value if Bragg reflections appear discontinuous or un-realistically choppy.",
    )
    sim_params_group.add_argument(
        "--xtalSize",
        type=float,
        default=None,
        help="Crystal size in mm. If not specified, Nabc will be used to determine crystal size.",
    )
    sim_params_group.add_argument(
        "--Nabc",
        default=[30,30,30],
        nargs=3,
        type=float,
        help="Number of unit cells along a, b, and c axes. Used to determine crystal size if --xtalSize is not set. Format: 'N_a N_b N_c'. Default is 30 30 30",
    )
    sim_params_group.add_argument(
        "--spotScale",
        default=None,
        help="Override the crystal size parameter to scale spot intensities directly. Float value. Useful for quick intensity adjustments. Also, for best results, use spotScale as opposed to xtalSize, as fine-tuning spotScale leads to more realistic intensities.",
        type=float,
    )
    sim_params_group.add_argument(
        "--ksol",
        default=0.4,
        type=float,
        help="Solvent scattering parameter (k_sol) for structure factor calculation. Default is 0.4.",
    )
    sim_params_group.add_argument(
        "--bsol",
        default=120,
        type=float,
        help="Solvent B-factor (b_sol) for structure factor calculation. Default is 120.",
    )
    sim_params_group.add_argument(
        "--waterThick",
        type=float,
        default=2.5,
        help="Thickness of water in mm for background simulation. Default is 2.5 mm.",
    )
    sim_params_group.add_argument(
        "--verbose", action="store_true", help="Enable verbose output during simulation."
    )

    # Rotation and Image Control Group
    rot_img_group = parser.add_argument_group("Rotation and Image Control")
    rot_img_group.add_argument(
        "--totalDeg",
        type=float,
        help="Total amount of crystal rotation in degrees. Default is 180 degrees.",
        default=180,
    )
    rot_img_group.add_argument(
        "--numimg",
        type=int,
        default=180,
        help="Number of images to simulate over the total rotation range. Default is 180 images for 180 degrees, meaning 1 degree per image.",
    )
    rot_img_group.add_argument(
        "--rotate",
        action="store_true",
        help="If set, the crystal will be rotated continuously between exposures, simulating a smeared image within each step (e.g., for rotation method data). If not set, the crystal is static during each exposure.",
    )
    rot_img_group.add_argument(
        "--numPhiSteps",
        type=int,
        default=10,
        help="Number of mini-simulations to perform between phi and phi+delta_phi if --rotate is enabled. Higher values give smoother smearing but increase simulation time significantly.",
    )
    rot_img_group.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run number to be included in output filenames (e.g., shot_R_00001.cbf). Useful for organizing multiple simulation runs. Default is 1.",
    )
    rot_img_group.add_argument(
        "--testShot",
        action="store_true",
        help="Simulate only a single shot for quick testing purposes, regardless of --numimg.",
    )

    # Detector and Output Control Group
    det_output_group = parser.add_argument_group("Detector and Output Control")
    det_output_group.add_argument(
        "--dist",
        type=float,
        help="Detector distance in mm from the sample. Default is 200 mm.",
        default=200,
    )
    det_output_group.add_argument(
        "--gain",
        default=1,
        type=float,
        help="ADU (Area detector units) per photon. This is a scaling factor applied to the simulated photon counts. Default is 1.",
    )
    det_output_group.add_argument(
        "--calib",
        default=3,
        type=float,
        help="Detector calibration noise percentage. This option multiplies each simulated pixel by random noise term, whose strength is proportional to a percentage of the signal. Default is 3 (meaning 3 percent).",
    )
    det_output_group.add_argument(
        "--PSF",
        default=0,
        type=float,
        help="Detector Point Spread Function (PSF) FWHM (Full Width at Half Maximum) in mm. Simulates blurring due to detector response. Default is 0 (no PSF).",
    )
    det_output_group.add_argument(
        "--ADC",
        default=0,
        type=float,
        help="ADC (Analog-to-Digital Converter) offset in ADU. This is a baseline offset added to all pixels. Default is 0.",
    )
    det_output_group.add_argument(
        "--cbf",
        action="store_true",
        help="In addition to Nexus output, save a CBF file for each simulated image. The CBF file will lack goniometer information though and therefore will not work with XDS.",
    )
    det_output_group.add_argument(
        "--noWaveImg",
        action="store_true",
        help="Do not calculate the wavelength-per-pixel. This can speed up simulation and save disk space if per-pixel wavelength information is not needed.",
    )
    det_output_group.add_argument(
        "--eigerMask",
        action="store_true",
        help="if True, simulate onto an Eiger panel geometry. Pixels in gaps between Eiger panels will be set to -1.",
    )

    # Input Data and GPU Group
    input_gpu_group = parser.add_argument_group("Input Data and GPU Configuration")
    input_gpu_group.add_argument(
        "--specFile",
        default=None,
        type=str,
        help="""Path to a .lam file (Precoginition format) defining the incident beam spectrum (intensity vs. wavelength). If not provided, a monochromatic simulation will be used. To obtain a default spectrum similar to bioCars, try:
wget https://raw.githubusercontent.com/dermen/e080_laue/master/from_vukica.lam"""
    )
    input_gpu_group.add_argument(
        "--pdbFile",
        default=None,
        type=str,
        help="Path to a PDB file for structure factor simulation. The structure factors will be calculated from this PDB unless --mtzFile is provided.",
    )
    input_gpu_group.add_argument(
        "--mtzFile",
        default=None,
        type=str,
        help="Path to an MTZ file containing pre-calculated structure factors. This argument supersedes --pdbFile if provided, and is generally faster.",
    )
    input_gpu_group.add_argument(
        "--mtzLabel",
        default=None,
        type=str,
        help="MTZ label pointing to the structure factors within the MTZ file (e.g., 'F(+),SIGF(+)'). Required if --mtzFile is used.",
    )
    input_gpu_group.add_argument(
        "--nitro",
        action="store_true",
        help="Simulate nitrogenase-specific spread data. This option will override --pdbFile with a predefined nitrogenase PDB (datsimx/fcalcs/nitro.pdb).",
    )
    input_gpu_group.add_argument(
        "--rubre",
        action="store_true",
        help="Simulate rubredoxin-specific spread data. This option will override --pdbFile with a predefined rubredoxin PDB (datsimx/fcalcs/rubre.pdb).",
    )
    input_gpu_group.add_argument(
        "--useSpreadData",
        action="store_true",
        help="Enable the use of energy-dependent spread data (e.g., for anomalous scattering effects). Typically used in conjunction with --nitro or --rubre.",
    )
    input_gpu_group.add_argument(
        "--monoEnergy",
        type=float,
        default=7120,
        help="Perform a monochromatic simulation using this energy (in eV). Note, if specFile is provide, specFile will override monoEnergy.",
    )
    input_gpu_group.add_argument(
        "--ndev",
        type=int,
        default=1,
        help="Number of GPU devices per compute node to utilize for parallel simulation. Default is 1.",
    )
    input_gpu_group.add_argument(
        "--cuda",
        action="store_true",
        help="Set the DIFFBRAGG_USE_CUDA=1 environment variable to explicitly force running the CUDA kernel for simulations. Requires a CUDA-compatible GPU and DiffBragg compiled with CUDA support.",
    )

    # Crystal Splitting Group (for two-domain simulations)
    split_group = parser.add_argument_group("Crystal Splitting Parameters (for two-domain simulations)")
    split_group.add_argument(
        "--splitPhiStart",
        type=float,
        default=-1,
        help="Start of the rotation range (in degrees) where a second crystal domain will be simulated. Set to -1 to disable crystal splitting. (inclusive)",
    )
    split_group.add_argument(
        "--splitPhiEnd",
        type=float,
        default=-1,
        help="End of the rotation range (in degrees) where a second crystal domain will be simulated. Set to -1 to disable crystal splitting. (inclusive)",
    )
    split_group.add_argument(
        "--splitRotAxis",
        type=float,
        nargs =3,
        default=[1,0,0],
        help="Rotation axis (x,y,z comma-separated values, e.g., '1 0 0' or '0.5 0.5 0') for the second crystal domain. This rotation is applied *once* to the second crystal's initial orientation relative to the main crystal.",
    )
    split_group.add_argument(
        "--splitRotAngle",
        type=float,
        default=0.1,
        help="Rotation angle (in degrees) for the second crystal domain, applied about --splitRotAxis relative to the main crystal. Default is 0.1 degrees.",
    )
    split_group.add_argument(
        "--splitScale",
        type=float,
        default=0.5,
        help="Intensity scale factor (0 to 1) for the second crystal domain relative to the main crystal. For example, 0.5 means the second domain contributes half the intensity of the first. Default is 0.5.",
    )

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
    import dxtbx

    from datsimx import make_nexus
    from datsimx.fcalcs import fcalc, db_anom_inputs

    if args.cuda:
        os.environ["DIFFBRAGG_USE_CUDA"] = "1"

    total_rot = np.pi* args.totalDeg/180

    # convenience files from this repository
    this_dir = os.path.dirname(__file__)

    pdb_file = os.path.join(this_dir, '7lvc.pdb')
    if args.pdbFile is not None:
        pdb_file = args.pdbFile
    if args.nitro:
        pdb_file = os.path.join(this_dir, "fcalcs/nitro.pdb")
        assert os.path.exists(pdb_file)
        print(f"Will use nitro file {pdb_file}")
    if args.rubre:
        pdb_file = os.path.join(this_dir, "fcalcs/rubre.pdb")
        assert os.path.exists(pdb_file)
        print(f"Will use rubre file {pdb_file}")
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

    # Eiger model
    DETECTOR = DetectorFactory.simple(
        sensor='PAD',
        distance=args.dist,  # mm
        beam_centre=(155.5875, 163.6125),  # mm
        fast_direction='+x',
        slow_direction='-y',
        pixel_size=(.075, .075),  # mm
        image_size=(4148, 4362))
    MASK = np.ones((4362, 4148)).astype(bool)
    if args.eigerMask:
        mask_file = os.path.join(this_dir, "eiger_mask.hdf5")
        MASK = h5py.File(mask_file, "r")[args.maskKey][()]
        assert MASK.shape==(4362, 4148)

    if args.specFile is not None:
        spec_file = args.specFile
        try:
            weights, energies = db_utils.load_spectra_file(spec_file)
        except:
            weights, energies = db_utils.load_spectra_file(spec_file, delim=" ")

        if args.enSteps is not None and len(energies) > 1:
            from scipy.interpolate import interp1d
            wts_I = interp1d(energies, weights)# bounds_error=False, fill_value=0)
            energies = np.linspace(energies.min()+1e-6, energies.max()-1e-6, args.enSteps)
            weights = wts_I(energies)

        # should do a weighted mean here:
        ave_en = np.mean(energies)
        ave_wave = utils.ENERGY_CONV / ave_en
    else:
        energies = np.array([args.monoEnergy])
        weights = np.array([1])
        ave_en = args.monoEnergy
        ave_wave = utils.ENERGY_CONV / ave_en

    BEAM = BeamFactory.simple(ave_wave)

    print(f"fcalcs from {pdb_file}")
    fcalc_wave = None if args.useSpreadData else ave_wave
    no_anom_for_atoms = None
    if args.useSpreadData:
        no_anom_for_atoms = {"Fe"}
    Fcalc = fcalc.get_complex_fcalc_from_pdb(pdb_file, wavelength=ave_wave, k_sol=args.ksol, b_sol=args.bsol, no_anom_for_atoms=no_anom_for_atoms)

    spread_data = None
    if (args.nitro or args.rubre) and args.useSpreadData:
        # in this case leave Fcalc complex valued
        which = "nitro" if args.nitro else "rubre"
        atom_data, fprime, fdblprime = db_anom_inputs.gen_db_inputs(energies, which=which, rubre_fe_state=0)
        spread_data = {"atoms": atom_data, "fprime": fprime, "fdblprime": fdblprime}
    else:
        Fcalc = Fcalc.as_amplitude_array()

    water_bkgrnd = utils.sim_background(
        DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
        Fbg_vs_stol=None, sample_thick_mm=args.waterThick, density_gcm3=1, molecular_weight=18)

    air_Fbg, air_stol = np.loadtxt(air_name).T
    air_stol = flex.vec2_double(list(zip(air_Fbg, air_stol)))
    air = utils.sim_background(DETECTOR, BEAM, [ave_wave], [1], 
                            total_flux, pidx=0, beam_size_mm=beam_size_mm,
                            molecular_weight=14,
                            sample_thick_mm=5,
                            Fbg_vs_stol=air_stol, density_gcm3=1.2e-3)

    fdim, sdim = DETECTOR[0].get_image_size()
    img_sh = sdim, fdim
    water_bkgrnd = water_bkgrnd.as_numpy_array().reshape(img_sh)
    air = air.as_numpy_array().reshape(img_sh)


    num_en = len(energies)
    fluxes = weights / weights.sum() * total_flux * len(weights)
    print("Simulating with %d energies" % num_en)
    print("Mean energy:", ave_wave)
    sg = Fcalc.space_group()
    print("unit cell, space group:\n", Fcalc, "\n")

    ucell = Fcalc.unit_cell()
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

    # Initialize second crystal for splitting
    CRYSTAL2 = None
    if args.splitPhiStart != -1 and args.splitPhiEnd != -1:
        if COMM.rank == 0:
            print(f"Crystal splitting enabled from {args.splitPhiStart} to {args.splitPhiEnd} degrees.")
            print(f"Split crystal rotation: {args.splitRotAngle} degrees around {args.splitRotAxis}.")
            print(f"Split crystal intensity scale: {args.splitScale}.")

        # Create the second crystal by copying the first
        CRYSTAL2 = Crystal(a, b, c, sg)
        CRYSTAL2.set_U(randU.ravel()) # Start with the same orientation

        # Apply a small rotation to CRYSTAL2's U-matrix
        split_rot_ax = col(args.splitRotAxis)

        R_split = split_rot_ax.axis_and_angle_as_r3_rotation_matrix(args.splitRotAngle, deg=True)
        #split_rot_mat = Rotation.from_rotvec(np.deg2rad(args.splitRotAngle) * np.array(args.splitRotAxis)).as_matrix()
        U2_initial = sqr(CRYSTAL2.get_U()) * R_split
        CRYSTAL2.set_U(U2_initial)


    delta_phi = total_rot/ args.numimg

    gonio_axis = col((1,0,0))
    U0 = sqr(CRYSTAL.get_U())  # starting Umat
    if CRYSTAL2 is not None:
        U0_2 = sqr(CRYSTAL2.get_U())  # starting Umat2

    Nabc = args.Nabc

    mos_spread = args.mosSpread
    num_mos = args.mosDoms
    device_Id = COMM.rank % args.ndev
    tsims = []

    saved_h5s = []

    for i_shot in range(args.numimg):

        if i_shot % COMM.size != COMM.rank:
            continue
        tsim = time.time()
        print("Doing shot %d/%d" % (i_shot+1, args.numimg))
        current_phi_deg = delta_phi*i_shot*180/np.pi
        Rphi = gonio_axis.axis_and_angle_as_r3_rotation_matrix(delta_phi*i_shot, deg=False)
        Uphi = Rphi * U0
        CRYSTAL.set_U(Uphi)

        t = time.time()

        printout_pix=None
        from simtbx.diffBragg.device import DeviceWrapper
        with DeviceWrapper(device_Id) as _:

            verbose = False
            if args.verbose:
                verbose = COMM.rank==0
            out = diffBragg_forward(
                CRYSTAL, DETECTOR, BEAM, Fcalc, energies, fluxes,
                oversample=args.oversample, Ncells_abc=Nabc,
                mos_dom=num_mos, mos_spread=mos_spread, beamsize_mm=beam_size_mm,
                device_Id=device_Id,
                show_params=False, crystal_size_mm=args.xtalSize, printout_pix=printout_pix,
                verbose=verbose, default_F=0, interpolate=0,
                mosaicity_random_seeds=None, div_mrad=args.div,
                divsteps=args.divSteps,
                spot_scale_override=args.spotScale,
                show_timings=verbose,
                nopolar=False, diffuse_params=None,
                delta_phi=delta_phi*180/np.pi if args.rotate else None,
                num_phi_steps = args.numPhiSteps if args.rotate else 1,
                perpixel_wavelen=not args.noWaveImg,
                spread_data=spread_data)

        if args.noWaveImg:
            img = out
            wave_img = h_img = k_img = l_img = None
        else:
            img, wave_img, h_img, k_img, l_img = out

        # Simulate for the second crystal if within the split range
        img_split = np.zeros(img_sh, dtype=np.float32)
        if CRYSTAL2 and args.splitPhiStart <= current_phi_deg <= args.splitPhiEnd:
            Uphi2 = Rphi * U0_2 # Apply the same rotation to the second crystal's initial U-matrix
            CRYSTAL2.set_U(Uphi2)

            print(f"  --> Simulating split crystal for shot {i_shot+1} (phi={current_phi_deg:.2f} deg)")
            out2 = diffBragg_forward(
                CRYSTAL2, DETECTOR, BEAM, Fcalc, energies, fluxes, # Use same Fcalc, energies, fluxes
                oversample=args.oversample, Ncells_abc=Nabc,
                mos_dom=num_mos, mos_spread=mos_spread, beamsize_mm=beam_size_mm,
                device_Id=device_Id,
                show_params=False, crystal_size_mm=args.xtalSize, printout_pix=printout_pix,
                verbose=verbose, default_F=0, interpolate=0,
                mosaicity_random_seeds=None, div_mrad=args.div,
                divsteps=args.divSteps,
                spot_scale_override=args.spotScale,
                show_timings=verbose,
                nopolar=False, diffuse_params=None,
                delta_phi=delta_phi*180/np.pi if args.rotate else None,
                num_phi_steps = args.numPhiSteps if args.rotate else 1,
                perpixel_wavelen=False, # Only want wave_img from the main crystal
                spread_data=spread_data)
            img2 = out2

            #if len(img2.shape)==3:
            #    img2 = img2[0]
            img_split = img2 * args.splitScale # Apply scaling to the split component

        img = img + img_split # Combine images from both crystals 


        t = time.time()-t
        print("Took %.4f sec to sim" % t)
        if len(img.shape)==3:
            img = img[0]
            if wave_img is not None:
                wave_img = wave_img[0]
                h_img = h_img[0]
                k_img = k_img[0]
                l_img = l_img[0]

        bg_only = water_bkgrnd+air

        img_with_bg = img +water_bkgrnd + air

        SIM = nanoBragg(detector=DETECTOR, beam=BEAM)
        SIM.beamsize_mm = beam_size_mm
        SIM.exposure_s = 1
        SIM.flux = total_flux
        SIM.adc_offset_adu = args.ADC
        SIM.detector_psf_kernel_radius_pixels = 5
        SIM.detector_calibration_noice_pct = args.calib
        SIM.detector_psf_fwhm_mm = args.PSF
        SIM.quantum_gain = args.gain
        SIM.readout_noise_adu = 3
        SIM.raw_pixels += flex.double((img_with_bg).ravel())
        SIM.add_noise()
        if args.cbf:
            cbf_name = os.path.join(args.outdir, "shot_%d_%05d.cbf" % (args.run, i_shot+1))
            SIM.to_cbf(cbf_name, cbf_int=True)
        
        noise_image = SIM.raw_pixels.as_numpy_array()
        noise_image[ ~MASK] = -1
        
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
        if CRYSTAL2 is not None:
            if args.splitPhiStart <= current_phi_deg <= args.splitPhiEnd:
                h.create_dataset("Umat_split", data=CRYSTAL2.get_U())
            else:
                h.create_dataset("Umat_split", data=(1,0,0,0,1,0,0,0,1))

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
