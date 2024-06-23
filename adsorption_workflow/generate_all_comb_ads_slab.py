from pymatgen.transformations.advanced_transformations import (
    EnumerateStructureTransformation,
)
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from asimtools.utils import get_atoms,get_images

import numpy as np

import itertools
from math import comb

from ase import Atoms
from ase.constraints import FixedLine

from tqdm import tqdm

import os

from autocat.adsorption import place_adsorbate

from ase.db import connect



def generate_all_comb_ads_slab(
    slab_atoms_image, adsorbate, ads_slab_image, fixed_line=False
):

    def get_ads_xy_positions(
        ads_slab_image=None,  
    ):
        ads_slab_lst = get_images(**ads_slab_image)

        ads_slab_energy = [
            ads_slab.get_potential_energy() for ads_slab in ads_slab_lst
        ]
        ads_slab_energy_min_idx = np.argsort(ads_slab_energy)[0]

        ads_slab_most_stable = ads_slab_lst[ads_slab_energy_min_idx]
        ads_x, ads_y = ads_slab_most_stable.positions[-1, :-1]

        return {"positions":[float(np.round(ads_x,decimals=2)), float(np.round(ads_y,decimals=2))]}

    offset_xy = get_ads_xy_positions(ads_slab_image)["positions"]
    slab_atoms = get_atoms(**slab_atoms_image)
    supercell_size = slab_atoms_image["repeat"]

    num_sites = int(np.prod(supercell_size))

    atoms_prototype = (
        Atoms("Li", [(0, 0, 1.23 / 2)], cell=[4.33, 4.33, 1.23], pbc=[1, 1, 0])
        * supercell_size
    )
    atoms_prototype_frac_coord = atoms_prototype.get_scaled_positions()[:, :-1]


    for num_adsorbate in tqdm(range(1, num_sites)):

        adsorbate_ratio = np.round(num_adsorbate / num_sites, decimals=4)
        ads_slab_lst = []
        # ads_slab_lst = [0 for _ in range(999)]
        idx = 0
        total_combination = comb(num_sites, num_adsorbate)
        print(
            "Total combination: ",
            total_combination,
            f". With num_ads/num_sites: {num_adsorbate}/{num_sites}.",
        )
        ads_slab_db_bank_ratio_path = f"ads.{adsorbate}.conc.{adsorbate_ratio}"
        os.makedirs(ads_slab_db_bank_ratio_path)

        for combinations in itertools.combinations(range(num_sites), num_adsorbate):

            ads_frac_coord_lst = atoms_prototype_frac_coord[combinations, :]
            ads_cart_coord_lst = ads_frac_coord_lst * np.array(
                slab_atoms.cell.cellpar()[:2]
            )

            ads_slab_atoms = slab_atoms.copy()
            for ads_cart_coord in ads_cart_coord_lst:
                ads_slab_atoms = place_adsorbate(
                    ads_slab_atoms,
                    adsorbate=Atoms(adsorbate),
                    adsorption_site=ads_cart_coord + offset_xy,
                )

            if fixed_line:
                fix_atoms = ads_slab_atoms.constraints
                fix_line = FixedLine(
                    indices=list(
                        range(
                            len(ads_slab_atoms) - len(ads_cart_coord_lst),
                            len(ads_slab_atoms),
                        )
                    ),
                    direction=[0, 0, 1],
                )
                ads_slab_atoms.set_constraint([fix_line] + fix_atoms)

            ads_slab_lst.append(ads_slab_atoms)
            if len(ads_slab_lst) > 100:
                ads_slab_db_path = f"{ads_slab_db_bank_ratio_path}/{idx}.db"
                with connect(ads_slab_db_path) as ads_db:
                    for atoms in ads_slab_lst:
                        ads_db.write(
                            atoms,
                            adsorbate=adsorbate,
                            adsorbate_ratio=adsorbate_ratio,
                            num_adsorbate=int(num_adsorbate),
                            ads_site="all_comb",
                            size=".".join([str(i) for i in supercell_size]),
                        )
                ads_slab_lst = []
                idx += 1

        if len(ads_slab_lst) != 0:
            ads_slab_db_path = f"{ads_slab_db_bank_ratio_path}/{idx}.db"
            with connect(ads_slab_db_path) as ads_db:
                for atoms in ads_slab_lst:
                    ads_db.write(
                        atoms,
                        adsorbate=adsorbate,
                        adsorbate_ratio=adsorbate_ratio,
                        num_adsorbate=int(num_adsorbate),
                        ads_site="all_comb",
                        size=".".join([str(i) for i in supercell_size]),
                    )
        else:
            idx -= 1
        print("No. of db saved: ", idx + 1)

    return {}
