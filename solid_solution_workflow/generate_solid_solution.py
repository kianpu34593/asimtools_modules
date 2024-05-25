from pymatgen.core.structure import Structure  # , Element
from pymatgen.io.ase import AseAtomsAdaptor

from ase.db import connect
import os
import numpy as np
import re
from tqdm import tqdm

from ase.io import write as ase_write

from pymatgen.transformations.advanced_transformations import (
    EnumerateStructureTransformation,
)

from math import comb

from asimtools.utils import get_atoms


def generate_solid_solution(
        bulk_image,
        supercell_size, 
        subs_element,
        save_to_db: bool = True,
        ):
    AAA = AseAtomsAdaptor()
    enumlib_trans = EnumerateStructureTransformation(min_cell_size=1, max_cell_size=1)

    initial_struct = get_atoms(**bulk_image)
    base_struc = initial_struct.get_primitive_structure().make_supercell(supercell_size)

    base_struc_formula = base_struc.formula
    reduced_formula = base_struc.reduced_formula
    supercell_size_str = "".join([str(i) for i in supercell_size])

    parts = re.findall(r"([A-Z][a-z]*)(\d*)", base_struc_formula)
    orig_element, count = parts[0]

    db_path = f"{reduced_formula}_{supercell_size_str}_subs_{subs_element}.db"

    for i in range(int(count) - 1):
        max_combination_num = comb(int(count), i + 1)
        subs_element_conc = str(np.round((i + 1) / int(count), decimals=4))
        orig_element_conc = str(1 - float(subs_element_conc))
        enum_struc = base_struc.copy()
        enum_struc[orig_element] = "".join(
            [orig_element, orig_element_conc, subs_element, subs_element_conc]
        )
        ss = enumlib_trans.apply_transformation(
            enum_struc, return_ranked_list=max_combination_num
        )
        ss_atoms_lst = [AAA.get_atoms(d["structure"]) for d in ss]
        with connect(db_path) as db:
            for atoms in ss_atoms_lst:
                db.write(atoms,subs_conc=float(subs_element_conc))
        print(
            f"@ {subs_element}{subs_element_conc} total No. of configurations:",
            max_combination_num,
        )
        print(
            f"@ {subs_element}{subs_element_conc} inequivalent No. of configurations:",
            len(ss_atoms_lst),
        )
    return {}


if __name__ == "__main__":
    super_cell_size = [2, 2, 2]
    substitute_atom = "Cu"
    GenSolidSolution(super_cell_size, substitute_atom)
