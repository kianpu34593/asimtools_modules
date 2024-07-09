from pymatgen.transformations.advanced_transformations import (
    EnumerateStructureTransformation,
)
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.io import read
import numpy as np
from ase.utils.structure_comparator import SymmetryEquivalenceCheck

import itertools
from math import comb

from ase.db import connect
from ase.constraints import FixedLine

from autocat.adsorption import place_adsorbate

from asimtools.utils import get_atoms,get_images

from tqdm import tqdm

import time

import os
import re



def generate_inequivalent_ads_slab(
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
    
    def basis_transformation(cell_xy):
        x_norm_vec = cell_xy[0,:]/np.linalg.norm(cell_xy[0,:])
        y_norm_vec = cell_xy[1,:]/np.linalg.norm(cell_xy[1,:])
        transformed_mat = np.array([x_norm_vec,y_norm_vec]).T
        return transformed_mat

    def prepare_atoms_prototype(supercell_size, cell_xy, slab_atoms, offset_xy_transformed_frac):
        atoms_prototype = (Atoms("Li", [(0, 0, 1.23 / 2)], cell=[4.33, 4.33, 1.23], pbc=[1, 1, 0]) * supercell_size)
        atoms_prototype_cell = np.array(atoms_prototype.cell)
        atoms_prototype_frac_coord = atoms_prototype.get_scaled_positions()[:, :-1]
        atoms_prototype_cell[:2, :2] = cell_xy
        atoms_prototype.set_cell(atoms_prototype_cell, scale_atoms=True)
        atoms_prototype_frac_coord_offseted_transformed = atoms_prototype_frac_coord + offset_xy_transformed_frac
        atoms_prototype_frac_coord_offseted = np.dot(transformation_mat, atoms_prototype_frac_coord_offseted_transformed.T)
        all_ads_cart_coord = (slab_atoms.cell.cellpar()[:2].reshape(2, 1) * atoms_prototype_frac_coord_offseted).T
        all_ads_cart_coord = np.append(all_ads_cart_coord, (np.ones(all_ads_cart_coord.shape[0]) * 1.23 / 2).reshape(all_ads_cart_coord.shape[0], 1), axis=1)
        atoms_prototype.set_positions(all_ads_cart_coord)
        return atoms_prototype

    def save_to_database(ads_slab_lst, adsorbate, adsorbate_ratio, num_adsorbate, supercell_size, db_path):
        with connect(db_path) as ads_db:
            for atoms in ads_slab_lst:
                ads_db.write(
                    atoms,
                    adsorbate=adsorbate,
                    adsorbate_ratio=adsorbate_ratio,
                    num_adsorbate=int(num_adsorbate),
                    ads_site="inequivalent",
                    size=".".join([str(i) for i in supercell_size]),
                    #degeneracy=int(degeneracy)
                )

    offset_xy = get_ads_xy_positions(ads_slab_image)["positions"]
    slab_atoms = get_atoms(**slab_atoms_image)
    supercell_size = slab_atoms_image["repeat"]

    num_sites = int(np.prod(supercell_size))

    # indices = np.arange(0,num_sites)

    cell_xy = slab_atoms.cell[:2,:2]

    transformation_mat = basis_transformation(cell_xy)
    transformation_mat_inv = np.linalg.inv(transformation_mat)

    offset_xy_transformed = np.dot(transformation_mat_inv,offset_xy)
    
    offset_xy_transformed_frac=offset_xy_transformed/slab_atoms.cell.cellpar()[:2]

    atoms_prototype = prepare_atoms_prototype(supercell_size, cell_xy, slab_atoms, offset_xy_transformed_frac)

    for num_adsorbate in tqdm(range(1, num_sites)):
        adsorbate_ratio = np.round(num_adsorbate / num_sites, decimals=4)
        
        # # ads_slab_lst = [0 for _ in range(999)]
        # idx = 0
        total_combination = comb(num_sites, num_adsorbate)
        # all_atoms_lst = [atoms_prototype.copy() for _ in range(total_combination)]
        # print(
        #     "Total combination: ",
        #     total_combination,
        #     f". With num_ads/num_sites: {num_adsorbate}/{num_sites}.",
        # )
        # ads_slab_db_bank_ratio_path = f"ads.{adsorbate}.conc.{adsorbate_ratio}"
        # os.makedirs(ads_slab_db_bank_ratio_path) 
        # for i,combinations in enumerate(itertools.combinations(range(num_sites), num_adsorbate)):
        #     temp_atoms = all_atoms_lst[i]
        #     filtered_indices = indices[~np.isin(range(0,num_sites), list(combinations))]
        #     temp_atoms.symbols[filtered_indices] = 'H'
        #     # all_atoms_lst[i]=temp_atoms

        base_struc = AseAtomsAdaptor.get_structure(atoms_prototype)

        enum_struc = base_struc.copy()
        enum_struc['Li'] = "".join(
            [adsorbate, str(adsorbate_ratio), 'H', str(1-adsorbate_ratio)]
        )

        enumlib_trans = EnumerateStructureTransformation(min_cell_size=1, max_cell_size=1)
        ss = enumlib_trans.apply_transformation(
            enum_struc, return_ranked_list=total_combination
        )
        ss_atoms_lst = [AseAtomsAdaptor.get_atoms(d["structure"]) for d in ss]


        # comp = SymmetryEquivalenceCheck(to_primitive=True)

        
        # inequivalent_lst = [0 for _ in range(len(ss_atoms_lst))]
        # total_degeneracy = 0

        # for i,enumlib_atoms in tqdm(enumerate(ss_atoms_lst),total=len(ss_atoms_lst)):
        #     enumlib_atoms.wrap()
        #     results_lst = []
        #     for atoms in all_atoms_lst:
        #         results_lst.append(comp.compare(atoms, enumlib_atoms))
        #     #degeneracy = sum(comp.compare(atoms, enumlib_atoms) for atoms in all_atoms_lst)
        #     degeneracy = sum(results_lst)
   
        #     total_degeneracy += degeneracy
        #     inequivalent_lst[i]=(enumlib_atoms, degeneracy)
        #     indices_to_delete = np.where(results_lst)[0]
        #     for index in sorted(indices_to_delete, reverse=True):
        #         del all_atoms_lst[index]
        # print("Total degeneracy: ", total_degeneracy)

        ads_slab_lst = []

        # Record the start time
        start_time = time.time()
        for enumlib_atoms in ss_atoms_lst:
            ads_cart_coord_lst = enumlib_atoms.get_positions()[np.array(enumlib_atoms.get_chemical_symbols()) == adsorbate,:2].tolist()
            ads_slab_atoms = slab_atoms.copy()
            for ads_cart_coord in ads_cart_coord_lst:
                ads_slab_atoms = place_adsorbate(
                    ads_slab_atoms,
                    adsorbate=Atoms(adsorbate),
                    adsorption_site=ads_cart_coord
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
            ads_slab_atoms.wrap()
            ads_slab_lst.append(ads_slab_atoms)
    
        ads_slab_db_path = "enumlib.inequivalent.db"
        save_to_database(ads_slab_lst, adsorbate, adsorbate_ratio, num_adsorbate, supercell_size, ads_slab_db_path)

    return {}

if __name__ == "__main__":
    slab_atoms_image={"image_file":"/scratch/venkvis_root/venkvis/kianpu/projects/anodefree/production_enumlib/Cu_mp-30/100_ads_chained_workflow/step-1/Cu.100.id.0.all_sites.db",
                        "index": 0,
                        "repeat": [4,4,1],
    }
    adsorbate = 'Li'
    ads_slab_image = {"image_file":"/scratch/venkvis_root/venkvis/kianpu/projects/anodefree/production_enumlib/Cu_mp-30/100_ads_chained_workflow/step-1/Cu.100.id.0.ads.Li.all_sites.db"} 

    generate_inequivalent_ads_slab(slab_atoms_image,adsorbate,ads_slab_image)