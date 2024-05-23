from asimtools.asimmodules.geometry_optimization.atom_relax import atom_relax
from asimtools.utils import get_atoms, get_images
from ase.db import connect
import numpy as np
import re

def adsorption_energy(
    calc_id_slab,
    calc_id_bulk,
    ads_slab_image,
    num_adsorbate_mode,
    placement_mode=None, #all_sites or most_stable_site or ads_convergence
    adsorbate_image=None,
    adsorbate_energy=None,
    clean_slab_image=None,
    clean_slab_energy=None,
    update_db=True,
    skip_relaxed=False,
    fmax=0.01,
):
    if clean_slab_energy is None:
        if placement_mode == 'most_stable_site' or placement_mode == 'ads_convergence':
            clean_slab_image['index'] = ads_slab_image['index']
        results = atom_relax(calc_id=calc_id_slab, image=clean_slab_image,fmax=fmax)
        clean_slab_energy = results["energy"]
    with open("results.txt", "a") as f:
        print("Clean Slab Energy (eV): ", clean_slab_energy,file=f)
    if adsorbate_energy is None:
        results = atom_relax(calc_id=calc_id_bulk, image=adsorbate_image,fmax=fmax)
        atoms = get_atoms(**adsorbate_image)
        adsorbate_energy = results["energy"] / len(atoms)
    with open("results.txt", "a") as f:
        print("Adsorbate Energy (eV/atom): ", adsorbate_energy,file=f)
    atoms_lst = get_images(**ads_slab_image)
    path_to_db = ads_slab_image['image_file']
    for i,atoms in enumerate(atoms_lst):
        if skip_relaxed:
            try:
                atoms.get_potential_energy()
                continue
            except:
                pass
        if ads_slab_image['index'] == ":":
            idx = i+1
        else:
            idx = int(ads_slab_image['index'])+1

        results = atom_relax(calc_id=calc_id_slab, image={"atoms":atoms},fmax=fmax,prefix=str(idx))
        ads_slab_energy = results["energy"]
        relax_atoms =  get_atoms(image_file=results["files"]["traj"])
        ads_slab_pos = relax_atoms.get_positions()[
            -1, :
        ]
        
        with connect(path_to_db) as ads_db:
            size_lst = ads_db.get(id=idx).size.split('.')
        num_sites = np.prod([int(i) for i in size_lst])
        if num_adsorbate_mode == "multiple":
            with connect(path_to_db) as ads_db:
                num_adsorbate = int(ads_db.get(id=idx).num_adsorbate)
        elif num_adsorbate_mode == "single":
            num_adsorbate = 1
        else:
            raise RuntimeError("`num_adsorbate` needs to be specified. Available options: `multiple` or `single`")
        ads_energy_per_site = np.round((ads_slab_energy - clean_slab_energy - int(num_adsorbate)*adsorbate_energy)/num_sites,decimals=4)
        with open("results.txt", "a") as f:
            print("Adsorption Energy per site (eV): ", ads_energy_per_site,file=f)
            print("Adsorbate positions: ", ads_slab_pos,file=f)
            print("No. of sites: ", num_sites,file=f)
            print("No. of adsorbate: ", num_adsorbate,file=f)
            print("Adsorbate conc.: ", np.round(int(num_adsorbate)/num_sites,decimals=4),file=f)
        if update_db:
            with connect(path_to_db) as ads_db:
                ads_db.update(
                    id=idx,
                    atoms=relax_atoms,
                    # site_coords=ads_slab_pos,
                    site_coord_x=ads_slab_pos[0],
                    site_coord_y=ads_slab_pos[1],
                    site_coord_z=ads_slab_pos[2],
                    ads_energy_per_site=ads_energy_per_site,
                )

    return {}
