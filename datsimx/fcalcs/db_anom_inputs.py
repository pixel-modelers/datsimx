import iotbx.pdb
import os
import numpy as np
from simtbx.diffBragg import utils
from scipy.interpolate import interp1d

# Simulate Fcalc iron contributions for nitrogenase or rubredoxin

def gen_db_inputs(ens, which="nitro", rubre_fe_state=0):
    """
    ens: list of energies to compute fprime and fdblprime
    generates the inputs needed for diffBragg to simulate  SPREAD data
    """
    assert which in {"nitro", "rubre"}
    assert rubre_fe_state in {0,1}
    dirname = os.path.dirname(__file__)
    if which=="nitro":
        pdb_file = os.path.join( dirname, "av1_highres_aom_w_fe_oxstates.pdb")
    elif which=="rubre":
        pdb_file = os.path.join( dirname,"1brf.pdb")

    red = os.path.join( dirname, "pf-rd-red_fftkk.out")
    ox = os.path.join( dirname, "pf-rd-ox_fftkk.out")
    assert os.path.exists(red)
    assert os.path.exists(ox)

    red_dat = np.loadtxt(red)
    ox_dat = np.loadtxt(ox)

    pdb_in = iotbx.pdb.input(pdb_file)
    xrs = pdb_in.xray_structure_simple()
    xrs.convert_to_isotropic()
    scs = xrs.scatterers()
    atom_data = []
    for s in scs:
        if s.element_symbol()=="Fe":
            if which=="nitro":
                lab = s.occupancy

                if lab in {0.2, 0.47, 0.46}:
                    # reduced
                    sp_id = 0
                elif lab in {0.3}:
                    # oxidized
                    sp_id = 1
                #else:
                #    raise NotImplementedError()
            elif which=="rubre":
                sp_id = rubre_fe_state

            x,y,z = s.site # fractional coords
            b = s.b_iso()  # b factor
            o = 1 # assume real occupancy is 1 for each atom
            print(s.label,x,y,z,b)
            atom_data.append((x,y,z,b,o, sp_id))

    fp0 = interp1d( red_dat[:,0], red_dat[:,1] ) (ens)
    fdp0 = interp1d( red_dat[:,0], red_dat[:,2] ) (ens)
    
    fp1 = interp1d( ox_dat[:,0], ox_dat[:,1] )  (ens)
    fdp1 = interp1d( ox_dat[:,0], ox_dat[:,2] ) (ens)

    # format everything for diffBragg inputs
    atom_data = tuple(map(list, zip(*atom_data)))
    if which=="nitro":
        fprimes = list(fp0) + list(fp1)
        fdblprimes = list(fdp0) + list(fdp1)
    elif which=="rubre":
        if rubre_fe_state==0:
            fprimes = list(fp0) 
            fdblprimes = list(fdp0)
        else:
            fprimes = list(fp1)
            fdblprimes = list(fdp1)

    return atom_data, fprimes, fdblprimes


if __name__=="__main__":
    lam_file = "gauss_100ev.lam"
    wt, ens = utils.load_spectra_file(lam_file, as_spectrum=False)
    db_inputs = gen_db_inputs(ens)

