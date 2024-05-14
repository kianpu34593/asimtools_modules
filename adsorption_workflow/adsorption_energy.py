from asimtools.asimmodules.geometry_optimization.atom_relax import atom_relax
from asimtools.utils import get_atoms, get_images
from ase.db import connect
import numpy as np

def adsorption_energy(
    calc_id_slab,
    calc_id_bulk,
    ads_slab_image,
    clean_slab_image=None,
    adsorbate_image=None,
    adsorbate_energy=None,
    slab_energy=None,
    path_to_db=None,
    fmax=0.01,
):
    if slab_energy is None:
        clean_slab_image
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
    for i,atoms in enumerate(atoms_lst):
        ads_slab_image = {"atoms":atoms}
        results = atom_relax(calc_id=calc_id_slab, image=ads_slab_image,fmax=fmax)
        ads_slab_energy = results["energy"]
        relax_atoms =  get_atoms(image_file=results["files"]["image"])
        ads_slab_pos = relax_atoms.get_positions()[
            -1, :
        ]
        with open("results.txt", "a") as f:
            print("Adsorption Energy (eV): ", np.round(ads_slab_energy - clean_slab_energy - adsorbate_energy,decimals=4), "Adsorbate positions: ", ads_slab_pos,file=f)
        if path_to_db is not None:
            ads_db = connect(path_to_db)
            ads_db.update(
                id=i+1,
                atoms=relax_atoms,
                # site_coords=ads_slab_pos,
                site_coord_x=ads_slab_pos[0],
                site_coord_y=ads_slab_pos[1],
                site_coord_z=ads_slab_pos[2],
                ads_energy=np.round(ads_slab_energy - clean_slab_energy - adsorbate_energy,decimals=4),
            )
    return {}
