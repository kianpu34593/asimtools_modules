from typing import Tuple, Dict, Optional
import numpy as np 
from asimtools.utils import (
    get_atoms,
)
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model import StructOptimizer
from chgnet.model.model import CHGNet

def chgnet_relax(
                image:Dict,
                CHGNetRelaxerDict:Dict,
                TM_lst:list,
                StructOptimizerDict:Dict = {}
                ):
    AAA = AseAtomsAdaptor()
    chgnet = CHGNet.load()
    relaxer = StructOptimizer(**StructOptimizerDict)
    atoms = get_atoms(**image)

    result = relaxer.relax(atoms=atoms, **CHGNetRelaxerDict)
    atoms = AAA.get_atoms(result['final_structure'])
    pot_e = atoms.get_potential_energy()
    chgnet_magmoms = atoms.get_magnetic_moments()
    atoms.write("chgnet_relax.traj")
    unique_symbols = np.unique(atoms.get_chemical_symbols())
    non_TM_lst = [i for i in unique_symbols if i not in TM_lst]
    for TM in TM_lst:
        TM_mask = np.array(atoms.get_chemical_symbols()) == TM
        TM_magmom = chgnet_magmoms[TM_mask]
        TM_magmom_mean = np.round(np.mean(TM_magmom),decimals=0)
        chgnet_magmoms[TM_mask] = TM_magmom_mean
    for TM in non_TM_lst:
        TM_mask = np.array(atoms.get_chemical_symbols()) == TM
        TM_magmom = chgnet_magmoms[TM_mask]
        TM_magmom_mean = np.round(np.mean(TM_magmom),decimals=0)
        chgnet_magmoms[TM_mask] = TM_magmom_mean
    atoms.set_initial_magnetic_moments(magmoms=chgnet_magmoms)
    atoms.write("chgnet_prediction.traj")
    return {}