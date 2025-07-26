import mmtbx.utils
import mmtbx.programs.fmodel
import iotbx.pdb
from cctbx.eltbx import henke

def get_complex_fcalc_from_pdb(
        pdb_file,
        wavelength=None,
        dmin=1,
        dmax=None,
        fft=True,
        k_sol=0.435, b_sol=46, show_pdb_summary=False, no_anom_for_atoms=None):
    """
    produce a structure factor from PDB coords, see mmtbx/programs/fmodel.py for formulation
    k_sol, b_sol form the solvent component of the Fcalc: Fprotein + k_sol*exp(-b_sol*s^2/4) (I think)
    """
    pdb_in = iotbx.pdb.input(pdb_file)
    xray_structure = pdb_in.xray_structure_simple()
    if show_pdb_summary:
        xray_structure.show_summary()
    xray_structure.convert_to_isotropic()
    for sc in xray_structure.scatterers():
        #if sc.element_symbol() == "Fe":
        #    sc.occupancy = 1
        if wavelength is not None:
            if no_anom_for_atoms is not None and sc.element_symbol() in no_anom_for_atoms:
                continue
            expected_henke = henke.table(sc.element_symbol()).at_angstrom(wavelength)
            sc.fp = expected_henke.fp()
            sc.fdp = expected_henke.fdp()
    phil = mmtbx.programs.fmodel.master_phil
    params = phil.extract()
    params.high_resolution = dmin
    params.low_resolution = dmax
    params.fmodel.k_sol = k_sol
    params.fmodel.b_sol = b_sol
    params.structure_factors_accuracy.algorithm = 'fft' if fft else 'direct'
    f_model = mmtbx.utils.fmodel_from_xray_structure(
        xray_structure=xray_structure,
        f_obs=None,
        add_sigmas=False,
        params=params).f_model
    f_model = f_model.generate_bijvoet_mates()

    return f_model
