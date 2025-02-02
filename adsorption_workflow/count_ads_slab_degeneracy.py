from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.io import read
import numpy as np
from ase.utils.structure_comparator import SymmetryEquivalenceCheck

import itertools
from math import comb

from ase.db import connect
from ase.constraints import FixedLine

from asimtools.utils import get_atoms,get_images

from tqdm import tqdm

import time

from glob import glob 

import os
import re

def count_ads_slab_degeneracy(single_ads_slab_image,enumlib_ads_slab_db_path,idx = None,image = None,image_traj_path=None,update_db = False):
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

    def prepare_atoms_prototype(supercell_size, cell_xy, cell_alpha_beta, offset_xy_transformed_frac):
        atoms_prototype = (Atoms("Li", [(0, 0, 1.23 / 2)], cell=[4.33, 4.33, 1.23], pbc=[1, 1, 0]) * supercell_size)
        atoms_prototype_cell = np.array(atoms_prototype.cell)
        atoms_prototype_frac_coord = atoms_prototype.get_scaled_positions()[:, :-1]
        atoms_prototype_cell[:2, :2] = cell_xy
        atoms_prototype.set_cell(atoms_prototype_cell, scale_atoms=True)
        atoms_prototype_frac_coord_offseted_transformed = atoms_prototype_frac_coord + offset_xy_transformed_frac
        atoms_prototype_frac_coord_offseted = np.dot(transformation_mat, atoms_prototype_frac_coord_offseted_transformed.T)
        all_ads_cart_coord = (cell_alpha_beta.reshape(2, 1) * atoms_prototype_frac_coord_offseted).T
        all_ads_cart_coord = np.append(all_ads_cart_coord, (np.ones(all_ads_cart_coord.shape[0]) * 1.23 / 2).reshape(all_ads_cart_coord.shape[0], 1), axis=1)
        atoms_prototype.set_positions(all_ads_cart_coord)
        return atoms_prototype

    def basis_transformation(cell_xy):
        x_norm_vec = cell_xy[0,:]/np.linalg.norm(cell_xy[0,:])
        y_norm_vec = cell_xy[1,:]/np.linalg.norm(cell_xy[1,:])
        transformed_mat = np.array([x_norm_vec,y_norm_vec]).T
        return transformed_mat

    def update_database(db_path, idx, degeneracy):
        with connect(db_path) as ads_db:
            ads_db.update(
                id = idx+1,
                degeneracy=int(degeneracy)
            )
    if idx is None and image is not None:
        current_path = os.getcwd()
        idx = int(current_path.split('/')[-1].split('__')[1])
        #idx = int(image['image_file'].split('/')[-1].split('_')[0])

    with connect(enumlib_ads_slab_db_path) as ads_slab_db:
        ads_slab_row = ads_slab_db.get(id=idx+1)
    try:
        ads_slab_row.energy
        atoms_traj_path = glob(os.path.join(os.path.dirname(image_traj_path),f'{idx+1}_atom_relax.traj'))[0]
        ads_slab_atoms = read(atoms_traj_path,index=0)
    except:
        ads_slab_atoms = ads_slab_row.toatoms()

    num_adsorbate = ads_slab_row.num_adsorbate
    adsorbate = ads_slab_row.adsorbate
    supercell_size = [int(i) for i in ads_slab_row.size.split('.')]
    num_sites = int(np.prod(supercell_size))
    indices = np.arange(0,num_sites)

    adsorbate_pos = ads_slab_atoms.get_positions()[-num_adsorbate:,:]
    cell_xyz = np.array(ads_slab_atoms.cell)
    cell_xyz[:, 2] = [0, 0, 1.23]
    adsorbate_pos[:,2] = 1.23 / 2

    offset_xy = get_ads_xy_positions(single_ads_slab_image)["positions"]
    cell_xy = cell_xyz[:2,:2]
    cell_alpha_beta = ads_slab_atoms.cell.cellpar()[:2]
    transformation_mat = basis_transformation(cell_xy)
    transformation_mat_inv = np.linalg.inv(transformation_mat)
    offset_xy_transformed = np.dot(transformation_mat_inv,offset_xy)
    offset_xy_transformed_frac=offset_xy_transformed/cell_alpha_beta

    atoms_prototype = prepare_atoms_prototype(supercell_size, cell_xy, cell_alpha_beta, offset_xy_transformed_frac)

    # # adsorbate_2dmodel = Atoms(f'{adsorbate}{num_adsorbate}', positions=adsorbate_pos)
    # # adsorbate_2dmodel.set_cell(cell_xyz,scale_atoms = False)
    # # adsorbate_2dmodel.pbc = [1,1,0]

    # atoms_prototype = (Atoms("Li", [(0, 0, 1.23 / 2)], cell=[4.33, 4.33, 1.23], pbc=[1, 1, 0]) * supercell_size)
    # atoms_prototype_cell = np.array(atoms_prototype.cell)
    # atoms_prototype_frac_coord = atoms_prototype.get_scaled_positions()[:, :-1]
    # atoms_prototype_cell[:2, :2] = cell_xyz[:2, :2]
    # atoms_prototype.set_cell(atoms_prototype_cell, scale_atoms=True)
    # print(adsorbate_pos.shape)

    idx_to_keep_lst = []
    for i in range(adsorbate_pos.shape[0]):
        diff = np.abs(atoms_prototype.get_positions() - adsorbate_pos[i,:])
        idx_temp = np.where((diff < 1e-3).all(axis=1))[0][0].tolist()
        idx_to_keep_lst.append(idx_temp)
    adsorbate_2dmodel = atoms_prototype.copy()
    filtered_indices = [i for i in indices if i not in idx_to_keep_lst]
    adsorbate_2dmodel.symbols[filtered_indices] = 'H'

    total_combination = comb(num_sites, num_adsorbate)
    all_atoms_lst = [atoms_prototype.copy() for _ in range(total_combination)]

    for i,combinations in enumerate(itertools.combinations(range(num_sites), num_adsorbate)):
        temp_atoms = all_atoms_lst[i]
        filtered_indices = indices[~np.isin(range(0,num_sites), list(combinations))]
        temp_atoms.symbols[filtered_indices] = 'H'
        # all_atoms_lst[i]=temp_atoms
    
    comp = SymmetryEquivalenceCheck(to_primitive=True)
    degeneracy = sum(comp.compare(atoms, adsorbate_2dmodel) for atoms in all_atoms_lst)
    #print(degeneracy)
    #update_database(enumlib_ads_slab_db_path, idx, degeneracy)
    return {"degeneracy":degeneracy}

if __name__ == "__main__": 
    enumlib_ads_slab_db_path = "/scratch/venkvis_root/venkvis/kianpu/projects/anodefree/production_enumlib/Cu_mp-30/100_ads_chained_workflow/step-4/enumlib.inequivalent.db"
    idx_lst = [8, 6, 3, 11, 17, 7, 2, 9, 10, 13, 14, 16, 1, 15]
    single_ads_slab_image = {"image_file":"/scratch/venkvis_root/venkvis/kianpu/projects/anodefree/production_enumlib/Cu_mp-30/100_ads_chained_workflow/step-1/Cu.100.id.0.ads.Li.all_sites.db"} 

    image_traj_path = "/scratch/venkvis_root/venkvis/kianpu/projects/anodefree/production_enumlib/Cu_mp-30/100_ads_chained_workflow/step-5/*/"
    for idx in idx_lst:
        count_ads_slab_degeneracy(single_ads_slab_image,enumlib_ads_slab_db_path,idx = idx,image = None,image_traj_path=image_traj_path,update_db=True)
    #[8, 6, 3, 11, 17, 7, 2, 9, 10, 13, 14, 16, 1, 15]