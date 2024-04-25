from asimtools.asimmodules.geometry_optimization.atom_relax import atom_relax
from asimtools.utils import get_atoms


def adsorption_energy(
    calc_id_slab,
    calc_id_bulk,
    ads_slab_image,
    clean_slab_image=None,
    adsorbate_image=None,
    adsorbate_energy=None,
    slab_energy=None,
    path_to_db=None,
    id_num=None,
):

    if slab_energy is None:
        results = atom_relax(calc_id=calc_id_slab, image=clean_slab_image)
        clean_slab_energy = results["energy"]
    if adsorbate_energy is None:
        results = atom_relax(calc_id=calc_id_bulk, image=adsorbate_image)
        atoms = get_atoms(**adsorbate_image)
        adsorbate_energy = results["energy"] / len(atoms)
    results = atom_relax(calc_id=calc_id_slab, image=ads_slab_image)
    ads_slab_energy = results["energy"]
    ads_slab_pos = get_atoms(image_file=results["files"]["image"]).get_positions()[
        -1, :
    ]

    if path_to_db is not None:
        ads_db = connect(path_to_db)
        ads_db.update(
            id=id_num,
            atoms=atoms,
            site_coords=ads_slab_pos,
            ads_energy=np.round(ads_slab_energy - clean_slab_energy - adsorbate_energy),
        )
    return {}
