
def main():

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="""Diffraction image simulator with per-shot jitter.
Simplified version of mx_simulate.py that adds per-shot randomization of
crystal parameters (unit cell, orientation, mosaicity, scale, B-factor).
Does not support: multi-PDB (pdbFiles), crystal splitting, waveImg, spread data.
        """,
    )
    parser.add_argument(
        "outdir", type=str,
        help="Output directory for simulated images and Nexus master file.",
    )

    # Simulation Parameters
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument("--mosSpread", type=float, default=0.1,
        help="Mosaic angular spread FWHM in degrees.")
    sim_group.add_argument("--mosDoms", type=int, default=150,
        help="Number of mosaic domains.")
    sim_group.add_argument("--blueSausage", action="store_true",
        help="Enable blue-sausage effect: each shot is composed of 3 slightly "
             "mis-oriented sub-lattices (sausages), creating a jagged punctate "
             "mosaic pattern instead of smooth mosaicity.")
    sim_group.add_argument("--sausageMisori", type=float, default=0.05,
        help="Sigma of angular mis-orientation between sausages in degrees.")
    sim_group.add_argument("--div", type=float, default=0,
        help="Beam divergence in mrad.")
    sim_group.add_argument("--divSteps", type=int, default=0,
        help="Number of divergence steps in x and y.")
    sim_group.add_argument("--enSteps", type=int, default=100,
        help="Number of spectrum samples.")
    sim_group.add_argument("--oversample", type=int, default=1,
        help="Pixel oversampling factor.")
    sim_group.add_argument("--xtalSize", type=float, default=None,
        help="Crystal size in mm. Overrides Nabc for sizing.")
    sim_group.add_argument("--Nabc", default=[30,30,30], nargs=3, type=float,
        help="Number of unit cells along a, b, c axes.")
    sim_group.add_argument("--spotScale", default=None, type=float,
        help="Override crystal size to scale spot intensities directly.")
    sim_group.add_argument("--ksol", default=0.4, type=float,
        help="Solvent k_sol parameter for Fcalc.")
    sim_group.add_argument("--bsol", default=120, type=float,
        help="Solvent B-factor (b_sol) for Fcalc.")
    sim_group.add_argument("--bfactor", default=None, type=float,
        help="Override all atom B-factors (Angstrom^2).")
    sim_group.add_argument("--waterThick", type=float, default=2.5,
        help="Water thickness in mm for background.")
    sim_group.add_argument("--verbose", action="store_true")
    sim_group.add_argument(
        "--fluxScale",
        type=float,
        default=1,
        help="Scale the default flux of 5e9  by this amount (default=1)",
    )

    # Rotation and Image Control
    rot_group = parser.add_argument_group("Rotation and Image Control")
    rot_group.add_argument("--totalDeg", type=float, default=180,
        help="Total crystal rotation in degrees.")
    rot_group.add_argument("--numimg", type=int, default=180,
        help="Number of images to simulate.")
    rot_group.add_argument("--static", action="store_true",
        help="Crystal fixed during exposure (no rotation blur).")
    rot_group.add_argument("--numPhiSteps", type=int, default=10,
        help="Sub-steps per frame for rotation blur.")
    rot_group.add_argument("--run", type=int, default=1,
        help="Run number for output filenames.")
    rot_group.add_argument("--testShot", action="store_true",
        help="Simulate only 1 image for quick testing.")

    # Detector and Output
    det_group = parser.add_argument_group("Detector and Output Control")
    det_group.add_argument("--dist", type=float, default=200,
        help="Detector distance in mm.")
    det_group.add_argument("--gain", default=1, type=float,
        help="ADU per photon.")
    det_group.add_argument("--calib", default=3, type=float,
        help="Calibration noise percentage.")
    det_group.add_argument("--PSF", default=0, type=float,
        help="Detector PSF FWHM in mm.")
    det_group.add_argument("--ADC", default=0, type=float,
        help="ADC offset in ADU.")
    det_group.add_argument("--pilatus", action="store_true",
        help="Use Pilatus 6M instead of Eiger.")
    det_group.add_argument("--panelMask", action="store_true",
        help="Apply detector panel gap mask.")
    det_group.add_argument("--noNoise", action="store_true",
        help="Skip noise; save background separately.")
    det_group.add_argument("--writeMtz", action="store_true",
        help="Write ground truth Fcalc as MTZ to outdir/ground_truth.mtz.")

    # Input Data and GPU
    input_group = parser.add_argument_group("Input Data and GPU")
    input_group.add_argument("--specFile", default=None, type=str,
        help="Path to .lam spectrum file. If not provided, monochromatic.")
    input_group.add_argument("--pdbFile", default=None, type=str,
        help="Path to PDB file for structure factors.")
    input_group.add_argument("--mtzFile", default=None, type=str,
        help="Path to MTZ file with pre-calculated structure factors.")
    input_group.add_argument("--mtzLabel", default=None, type=str,
        help="MTZ column label (e.g. 'F(+),SIGF(+)').")
    input_group.add_argument("--monoEnergy", type=float, default=7120,
        help="Monochromatic energy in eV (used if no specFile).")
    input_group.add_argument("--ndev", type=int, default=1,
        help="Number of GPU devices per node.")
    input_group.add_argument("--cuda", action="store_true",
        help="Force CUDA backend.")

    # Jitter Parameters
    jitt_group = parser.add_argument_group("Per-shot Jitter Parameters",
        "All jitter values are in PERCENT of the base value (except jitterRot which "
        "is in degrees). Set to 0 (default) to disable.")
    jitt_group.add_argument("--jitterUcell", type=float, default=0,
        help="Percent jitter for unit cell edge lengths (e.g. 0.1 = 0.1%% of a,b,c). "
             "Respects crystal symmetry: tetragonal/hexagonal jitter a(=b) and c "
             "independently; orthorhombic jitters a,b,c independently; etc.")
    jitt_group.add_argument("--jitterUcellAngle", type=float, default=0,
        help="Percent jitter for unit cell angles (monoclinic beta, or triclinic "
             "alpha/beta/gamma). E.g. 0.5 = 0.5%% of base angle.")
    jitt_group.add_argument("--jitterRot", type=float, default=0,
        help="Sigma in degrees for orientation jitter (absolute, not percent).")
    jitt_group.add_argument("--jitterMosSpread", type=float, default=0,
        help="Percent jitter for mosaic spread FWHM (e.g. 10 = 10%% of base mosSpread).")
    jitt_group.add_argument("--jitterScale", type=float, default=0,
        help="Percent jitter for intensity scale (e.g. 10 = 10%% variation around 1.0).")
    jitt_group.add_argument("--jitterNabc", type=float, default=0,
        help="Percent jitter for mosaic domain size Nabc (e.g. 10 = 10%% of base Na,Nb,Nc). "
             "Each axis jittered independently. Values clamped to >= 1.")
    jitt_group.add_argument("--jitterBfactor", type=float, default=0,
        help="Sigma in Angstrom^2 for per-shot B-factor perturbation (absolute, not "
             "percent, since base B-factor can be zero).")
    jitt_group.add_argument("--jitterDist", type=str, default="normal",
        choices=["normal", "uniform"],
        help="Distribution for jitter draws. 'normal': Gaussian with given sigma. "
             "'uniform': uniform over [-sigma*sqrt(3), +sigma*sqrt(3)] "
             "(same variance as the Gaussian, but bounded).")
    jitt_group.add_argument("--seed", type=int, default=0,
        help="Random seed for jitter reproducibility.")

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
    from cctbx.uctbx import unit_cell

    from datsimx import make_nexus
    from datsimx.fcalcs import fcalc

    if args.cuda:
        os.environ["DIFFBRAGG_USE_CUDA"] = "1"

    has_jitter = any([args.jitterUcell, args.jitterUcellAngle, args.jitterRot,
                      args.jitterMosSpread, args.jitterScale, args.jitterBfactor,
                      args.jitterNabc])

    # Jitter draw function: normal or uniform with matched variance
    sqrt3 = np.sqrt(3)
    def jdraw(rng, sigma, size=None):
        """Draw from normal or uniform distribution with the given sigma (std dev)."""
        if args.jitterDist == "uniform":
            half = sigma * sqrt3
            return rng.uniform(-half, half, size=size)
        else:
            return rng.normal(0, sigma, size=size)

    total_rot = np.pi * args.totalDeg / 180

    # convenience files
    this_dir = os.path.dirname(__file__)

    pdb_file = os.path.join(this_dir, '7lvc.pdb')
    if args.pdbFile is not None:
        pdb_file = args.pdbFile
    air_name = os.path.join(this_dir, 'air.stol')
    total_flux = 5e9 * args.fluxScale
    beam_size_mm = 0.01

    if COMM.rank == 0:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        cmdfile = args.outdir + "/commandline_run%d.txt" % args.run
        with open(cmdfile, "w") as o:
            o.write(" ".join(sys.argv) + "\n")
    COMM.barrier()

    # Detector setup
    if args.pilatus:
        DETECTOR = DetectorFactory.simple(
            sensor='PAD', distance=args.dist,
            beam_centre=(211.836, 217.322),
            fast_direction='+x', slow_direction='-y',
            pixel_size=(.172, .172), image_size=(2463, 2527))
        MASK = np.ones((2527, 2463)).astype(bool)
        if args.panelMask:
            mask_file = os.path.join(this_dir, "pilatus_mask.hdf5")
            MASK = h5py.File(mask_file, "r")["mask"][()]
    else:
        DETECTOR = DetectorFactory.simple(
            sensor='PAD', distance=args.dist,
            beam_centre=(155.5875, 163.6125),
            fast_direction='+x', slow_direction='-y',
            pixel_size=(.075, .075), image_size=(4148, 4362))
        MASK = np.ones((4362, 4148)).astype(bool)
        if args.panelMask:
            mask_file = os.path.join(this_dir, "eiger_mask.hdf5")
            MASK = h5py.File(mask_file, "r")["mask"][()]

    # Spectrum
    if args.specFile is not None:
        try:
            weights, energies = db_utils.load_spectra_file(args.specFile)
        except:
            weights, energies = db_utils.load_spectra_file(args.specFile, delim=" ")
        if args.enSteps is not None and len(energies) > 1:
            from scipy.interpolate import interp1d
            wts_I = interp1d(energies, weights)
            energies = np.linspace(energies.min()+1e-6, energies.max()-1e-6, args.enSteps)
            weights = wts_I(energies)
        ave_en = np.mean(energies)
        ave_wave = utils.ENERGY_CONV / ave_en
    else:
        energies = np.array([args.monoEnergy])
        weights = np.array([1])
        ave_en = args.monoEnergy
        ave_wave = utils.ENERGY_CONV / ave_en

    BEAM = BeamFactory.simple(ave_wave)

    # Fcalc
    if args.mtzFile is not None:
        from iotbx import mtz
        M = mtz.object(args.mtzFile)
        col_label = args.mtzLabel
        if col_label is None:
            raise ValueError("--mtzLabel required when using --mtzFile")
        ma = M.as_miller_arrays()
        Fcalc = None
        for arr in ma:
            if arr.info().label_string() == col_label:
                Fcalc = arr
                break
        if Fcalc is None:
            raise ValueError("Could not find label '%s' in MTZ" % col_label)
    else:
        print("fcalcs from %s" % pdb_file)
        Fcalc = fcalc.get_complex_fcalc_from_pdb(
            pdb_file, wavelength=ave_wave,
            k_sol=args.ksol, b_sol=args.bsol,
            b_factor=args.bfactor)
        Fcalc = Fcalc.as_amplitude_array()

    # Background
    water_bkgrnd = utils.sim_background(
        DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
        Fbg_vs_stol=None, sample_thick_mm=args.waterThick, density_gcm3=1, molecular_weight=18)

    air_Fbg, air_stol = np.loadtxt(air_name).T
    air_stol = flex.vec2_double(list(zip(air_Fbg, air_stol)))
    air = utils.sim_background(
        DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
        molecular_weight=14, sample_thick_mm=5,
        Fbg_vs_stol=air_stol, density_gcm3=1.2e-3)

    fdim, sdim = DETECTOR[0].get_image_size()
    img_sh = sdim, fdim
    water_bkgrnd = water_bkgrnd.as_numpy_array().reshape(img_sh)
    air = air.as_numpy_array().reshape(img_sh)
    bg_only = water_bkgrnd + air

    num_en = len(energies)
    fluxes = weights / weights.sum() * total_flux * len(weights)
    print("Simulating with %d energies" % num_en)
    sg = Fcalc.space_group()
    print("unit cell, space group:\n", Fcalc, "\n")

    # Crystal setup
    ucell_base = Fcalc.unit_cell()
    Breal = ucell_base.orthogonalization_matrix()
    a = Breal[0], Breal[3], Breal[6]
    b = Breal[1], Breal[4], Breal[7]
    c = Breal[2], Breal[5], Breal[8]
    CRYSTAL = Crystal(a, b, c, sg)

    # Determine crystal system from space group number for symmetry-aware ucell jitter
    # Free parameters per system:
    #   cubic:        a           (b=a, c=a, angles=90)
    #   tetragonal:   a, c        (b=a, angles=90)
    #   hexagonal:    a, c        (b=a, angles=90/90/120)
    #   trigonal:     a, c        (b=a, angles=90/90/120, hexagonal setting)
    #   orthorhombic: a, b, c     (angles=90)
    #   monoclinic:   a, b, c, beta  (alpha=gamma=90)
    #   triclinic:    a, b, c, alpha, beta, gamma
    from cctbx import sgtbx
    sg_number = sgtbx.space_group_type(sg).number()
    if sg_number <= 2:
        crystal_system = "triclinic"
    elif sg_number <= 15:
        crystal_system = "monoclinic"
    elif sg_number <= 74:
        crystal_system = "orthorhombic"
    elif sg_number <= 142:
        crystal_system = "tetragonal"
    elif sg_number <= 167:
        crystal_system = "trigonal"
    elif sg_number <= 194:
        crystal_system = "hexagonal"
    else:
        crystal_system = "cubic"
    if COMM.rank == 0:
        print("Crystal system: %s (SG #%d)" % (crystal_system, sg_number))

    randU = None
    if COMM.rank == 0:
        randU = Rotation.random(random_state=args.seed)
        randU = randU.as_matrix()
    randU = COMM.bcast(randU)
    CRYSTAL.set_U(randU.ravel())

    delta_phi = total_rot / args.numimg
    gonio_axis = col((1, 0, 0))
    U0 = sqr(CRYSTAL.get_U())

    Nabc = args.Nabc
    device_Id = COMM.rank % args.ndev
    tsims = []

    # Pre-compute d_star_sq for B-factor jitter
    if args.jitterBfactor > 0:
        d_star_sq_data = Fcalc.d_star_sq().data()
        Fcalc_base_data = Fcalc.data().deep_copy()

    h5_name = os.path.join(args.outdir, "shots_%d_rank%d.h5" % (args.run, COMM.rank))
    with h5py.File(h5_name, 'w') as h:

        for i_shot in range(args.numimg):
            if i_shot % COMM.size != COMM.rank:
                continue

            tsim = time.time()
            print("Doing shot %d/%d" % (i_shot + 1, args.numimg))

            rng = np.random.RandomState(args.seed + i_shot)

            # --- Goniometer rotation ---
            Rphi = gonio_axis.axis_and_angle_as_r3_rotation_matrix(delta_phi * i_shot, deg=False)
            Uphi = Rphi * U0

            # --- Jitter orientation ---
            if args.jitterRot > 0:
                rotvec = jdraw(rng, np.deg2rad(args.jitterRot), size=3)
                R_jitt = sqr(Rotation.from_rotvec(rotvec).as_matrix().ravel())
                Uphi = R_jitt * Uphi

            # --- Jitter unit cell (symmetry-aware, percent of base values) ---
            crystal_shot = CRYSTAL
            # params order: (a, b, c, alpha, beta, gamma)
            p = list(ucell_base.parameters())  # base values
            ucell_shot_params = list(p)
            if args.jitterUcell > 0 or args.jitterUcellAngle > 0:
                pctU = args.jitterUcell / 100.0
                pctA = args.jitterUcellAngle / 100.0
                if crystal_system == "cubic":
                    da = jdraw(rng, p[0] * pctU) if pctU else 0
                    ucell_shot_params[0] += da
                    ucell_shot_params[1] += da  # b=a
                    ucell_shot_params[2] += da  # c=a
                elif crystal_system in ("tetragonal", "hexagonal", "trigonal"):
                    da = jdraw(rng, p[0] * pctU) if pctU else 0
                    dc = jdraw(rng, p[2] * pctU) if pctU else 0
                    ucell_shot_params[0] += da
                    ucell_shot_params[1] += da  # b=a
                    ucell_shot_params[2] += dc
                elif crystal_system == "orthorhombic":
                    if pctU:
                        ucell_shot_params[0] += jdraw(rng, p[0] * pctU)
                        ucell_shot_params[1] += jdraw(rng, p[1] * pctU)
                        ucell_shot_params[2] += jdraw(rng, p[2] * pctU)
                elif crystal_system == "monoclinic":
                    if pctU:
                        ucell_shot_params[0] += jdraw(rng, p[0] * pctU)
                        ucell_shot_params[1] += jdraw(rng, p[1] * pctU)
                        ucell_shot_params[2] += jdraw(rng, p[2] * pctU)
                    if pctA:
                        ucell_shot_params[4] += jdraw(rng, p[4] * pctA)  # beta
                else:  # triclinic
                    if pctU:
                        ucell_shot_params[0] += jdraw(rng, p[0] * pctU)
                        ucell_shot_params[1] += jdraw(rng, p[1] * pctU)
                        ucell_shot_params[2] += jdraw(rng, p[2] * pctU)
                    if pctA:
                        ucell_shot_params[3] += jdraw(rng, p[3] * pctA)  # alpha
                        ucell_shot_params[4] += jdraw(rng, p[4] * pctA)  # beta
                        ucell_shot_params[5] += jdraw(rng, p[5] * pctA)  # gamma

                new_ucell = unit_cell(ucell_shot_params)
                Br = new_ucell.orthogonalization_matrix()
                a_j = Br[0], Br[3], Br[6]
                b_j = Br[1], Br[4], Br[7]
                c_j = Br[2], Br[5], Br[8]
                crystal_shot = Crystal(a_j, b_j, c_j, sg)
            crystal_shot.set_U(Uphi)

            # --- Jitter mosaic spread (percent of base) ---
            mos_shot = args.mosSpread
            if args.jitterMosSpread > 0:
                sigma_mos = args.mosSpread * args.jitterMosSpread / 100.0
                mos_shot = max(0.001, args.mosSpread + jdraw(rng, sigma_mos))

            # --- Jitter B-factor (modify Fcalc amplitudes) ---
            Fcalc_shot = Fcalc
            delta_B = 0.0
            if args.jitterBfactor > 0:
                delta_B = jdraw(rng, args.jitterBfactor)
                corr = flex.exp(-delta_B / 4.0 * d_star_sq_data)
                Fcalc_shot = Fcalc.customized_copy(data=Fcalc_base_data * corr)

            # --- Jitter Nabc (percent of base) ---
            Nabc_shot = list(Nabc)
            if args.jitterNabc > 0:
                pctN = args.jitterNabc / 100.0
                Nabc_shot = [max(1, n + jdraw(rng, n * pctN)) for n in Nabc]

            # --- Simulate ---
            from simtbx.diffBragg.device import DeviceWrapper

            sausage_rotvecs = None
            sausage_Umats = None

            if args.blueSausage:
                # Blue-sausage: simulate 3 mis-oriented sub-lattices per shot
                n_sausages = 3
                mos_per_sausage = max(1, args.mosDoms // n_sausages)
                sausage_rotvecs = np.zeros((n_sausages, 3))
                sausage_Umats = np.zeros((n_sausages, 9))
                img = np.zeros(img_sh, dtype=np.float64)

                for i_s in range(n_sausages):
                    # Draw a small random mis-orientation for this sausage
                    rotvec_s = rng.normal(0, np.deg2rad(args.sausageMisori), size=3)
                    R_s = sqr(Rotation.from_rotvec(rotvec_s).as_matrix().ravel())
                    U_s = R_s * Uphi
                    crystal_shot.set_U(U_s)

                    sausage_rotvecs[i_s] = rotvec_s
                    sausage_Umats[i_s] = list(U_s)

                    with DeviceWrapper(device_Id) as _:
                        verbose_s = args.verbose and COMM.rank == 0 and i_s == 0
                        img_s = diffBragg_forward(
                            crystal_shot, DETECTOR, BEAM, Fcalc_shot, energies, fluxes,
                            oversample=args.oversample, Ncells_abc=Nabc_shot,
                            mos_dom=mos_per_sausage, mos_spread=mos_shot,
                            beamsize_mm=beam_size_mm,
                            device_Id=device_Id,
                            show_params=False, crystal_size_mm=args.xtalSize,
                            printout_pix=None,
                            verbose=verbose_s, default_F=0, interpolate=0,
                            mosaicity_random_seeds=None, div_mrad=args.div,
                            divsteps=args.divSteps,
                            spot_scale_override=args.spotScale,
                            show_timings=verbose_s,
                            nopolar=False, diffuse_params=None,
                            delta_phi=delta_phi * 180 / np.pi if not args.static else None,
                            num_phi_steps=args.numPhiSteps if not args.static else 1,
                            perpixel_wavelen=False,
                            spread_data=None)
                    if len(img_s.shape) == 3:
                        img_s = img_s[0]
                    img += img_s

                # Restore base orientation for metadata
                crystal_shot.set_U(Uphi)
                if COMM.rank == 0 and i_shot == 0:
                    print("Blue-sausage: %d sub-lattices, %d mos_doms each, "
                          "misori sigma=%.4f deg" % (n_sausages, mos_per_sausage,
                                                      args.sausageMisori))

            else:
                with DeviceWrapper(device_Id) as _:
                    verbose = args.verbose and COMM.rank == 0
                    img = diffBragg_forward(
                        crystal_shot, DETECTOR, BEAM, Fcalc_shot, energies, fluxes,
                        oversample=args.oversample, Ncells_abc=Nabc_shot,
                        mos_dom=args.mosDoms, mos_spread=mos_shot,
                        beamsize_mm=beam_size_mm,
                        device_Id=device_Id,
                        show_params=False, crystal_size_mm=args.xtalSize,
                        printout_pix=None,
                        verbose=verbose, default_F=0, interpolate=0,
                        mosaicity_random_seeds=None, div_mrad=args.div,
                        divsteps=args.divSteps,
                        spot_scale_override=args.spotScale,
                        show_timings=verbose,
                        nopolar=False, diffuse_params=None,
                        delta_phi=delta_phi * 180 / np.pi if not args.static else None,
                        num_phi_steps=args.numPhiSteps if not args.static else 1,
                        perpixel_wavelen=False,
                        spread_data=None)

                if len(img.shape) == 3:
                    img = img[0]

            # --- Jitter scale ---
            scale_factor = 1.0
            if args.jitterScale > 0:
                scale_factor = max(0.01, 1 + jdraw(rng, args.jitterScale / 100.0))
                img = img * scale_factor

            # --- Add background + noise ---
            img_with_bg = img + bg_only

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
            SIM.raw_pixels += flex.double(img_with_bg.ravel())
            if not args.noNoise:
                SIM.add_noise()

            if args.noNoise:
                output_image = img
                if COMM.rank == 0 and i_shot == 0:
                    bg_h5_name = os.path.join(args.outdir, "background_%d.h5" % args.run)
                    bg_save = bg_only.copy()
                    bg_save[~MASK] = -1
                    with h5py.File(bg_h5_name, "w") as bg_h:
                        bg_h.create_dataset("background", data=bg_save,
                                            compression="gzip", compression_opts=4, shuffle=True)
            else:
                output_image = SIM.raw_pixels.as_numpy_array()
            output_image[~MASK] = -1

            SIM.free_all()
            del SIM

            # --- Save per-shot data ---
            shot_name = "shot_%d_%05d" % (args.run, i_shot + 1)
            h.create_dataset(f"sim_image/{shot_name}", data=output_image.astype(np.float32),
                             dtype=np.float32, compression="gzip", compression_opts=4, shuffle=True)
            h.create_dataset(f"delta_phi/{shot_name}", data=delta_phi)
            h.create_dataset(f"Umat/{shot_name}", data=crystal_shot.get_U())
            h.create_dataset(f"Bmat/{shot_name}", data=crystal_shot.get_B())
            h.create_dataset(f"mos_spread/{shot_name}", data=mos_shot)
            if has_jitter:
                h.create_dataset(f"jitter/ucell_params/{shot_name}", data=ucell_shot_params)
                h.create_dataset(f"jitter/Nabc/{shot_name}", data=Nabc_shot)
                h.create_dataset(f"jitter/scale_factor/{shot_name}", data=scale_factor)
                h.create_dataset(f"jitter/delta_bfactor/{shot_name}", data=delta_B)
            if args.blueSausage and sausage_rotvecs is not None:
                h.create_dataset(f"sausage/rot_vecs_deg/{shot_name}",
                                 data=np.rad2deg(sausage_rotvecs))
                h.create_dataset(f"sausage/Umats/{shot_name}",
                                 data=sausage_Umats)

            tsim = time.time() - tsim
            if COMM.rank == 0:
                print("TSIM=%f" % tsim)
            tsims.append(tsim)
            if args.testShot:
                break

    tsims = COMM.reduce(tsims)
    if COMM.rank == 0:
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

        if args.blueSausage:
            import glob
            glob_s = os.path.join(args.outdir, f"shots_{args.run}_*.h5")
            rank_files = glob.glob(glob_s)
            shot_entries = []
            for rf in rank_files:
                with h5py.File(rf, 'r') as hf:
                    if 'sausage' in hf:
                        for sn in hf['sausage/rot_vecs_deg'].keys():
                            shot_entries.append((rf, sn))
            shot_entries.sort(key=lambda x: int(x[1].split("_")[-1]))

            sausage_file = os.path.join(args.outdir, f"sausage_run{args.run}.h5")
            nshots = len(shot_entries)
            with h5py.File(sausage_file, 'w') as sf:
                sf.attrs["description"] = (
                    "Blue-sausage ground truth: 3 sub-lattice orientations per shot. "
                    "rot_vecs_deg[i] = (3,3) mis-orientation rotation vectors (degrees) "
                    "applied to base crystal orientation for shot i. "
                    "Umats[i] = (3,9) resulting U matrices (flattened 3x3). "
                    "Umat_base[i] = (9,) base crystal U matrix before sausage perturbation.")
                sf.attrs["sausageMisori_deg"] = args.sausageMisori
                sf.attrs["n_sausages"] = 3
                rotvecs = np.zeros((nshots, 3, 3))
                umats = np.zeros((nshots, 3, 9))
                ubase = np.zeros((nshots, 9))
                for i, (rf, sn) in enumerate(shot_entries):
                    with h5py.File(rf, 'r') as hf:
                        rotvecs[i] = hf[f'sausage/rot_vecs_deg/{sn}'][()]
                        umats[i] = hf[f'sausage/Umats/{sn}'][()]
                        ubase[i] = hf[f'Umat/{sn}'][()]
                sf.create_dataset("rot_vecs_deg", data=rotvecs,
                                  compression="gzip", compression_opts=4)
                sf.create_dataset("Umats", data=umats,
                                  compression="gzip", compression_opts=4)
                sf.create_dataset("Umat_base", data=ubase,
                                  compression="gzip", compression_opts=4)
            print(f"Wrote sausage ground truth -> {sausage_file}")

        if args.writeMtz:
            mtz_path = os.path.join(args.outdir, "ground_truth.mtz")
            Fcalc_out = Fcalc
            if not Fcalc_out.is_xray_amplitude_array():
                Fcalc_out = Fcalc_out.as_amplitude_array()
            Fcalc_out.as_mtz_dataset(column_root_label='F').mtz_object().write(mtz_path)
            print(f"Wrote ground truth MTZ -> {mtz_path}")


if __name__ == "__main__":
    main()
