from asimtools.utils import (
    get_atoms,
)
import numpy as np 

def assign_init_magmom(
    atoms_image,
    TM_lst,
):
    atoms = get_atoms(**atoms_image)
    magmom_arr = atoms.get_magnetic_moments()
    unique_symbols = np.unique(atoms.get_chemical_symbols())
    non_TM_lst = [i for i in unique_symbols if i not in TM_lst]
    for TM in TM_lst:
        TM_mask = np.array(atoms.get_chemical_symbols()) == TM
        TM_magmom = magmom_arr[TM_mask]
        TM_magmom_mean = np.round(np.mean(TM_magmom),decimals=0)
        magmom_arr[TM_mask] = TM_magmom_mean
    for non_TM in non_TM_lst:
        non_TM_mask = np.array(atoms.get_chemical_symbols()) == non_TM
        non_TM_magmom = magmom_arr[non_TM_mask]
        non_TM_magmom_mean = np.round(np.mean(non_TM_magmom),decimals=0)
        magmom_arr[non_TM_mask] = non_TM_magmom_mean
    atoms.set_initial_magnetic_moments(magmoms=magmom_arr)
    atoms.write("atoms_init_magmom.traj")
    return {}