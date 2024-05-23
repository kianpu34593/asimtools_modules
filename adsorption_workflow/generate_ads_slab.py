from autocat import adsorption
from ase.db import connect 
from pathlib import Path
import matplotlib.pyplot as plt 
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import plot_slab as plot_slab_pymatgen
from asimtools.utils import get_atoms, get_images
import numpy as np
import os

def generate_ads_slab(
    slab_image,
    adsorbate_lst: list,  # one adsorbate e.g. Li
    placement_mode: str, # all_sites, most_stable_site (slab_image given a database of slab with adsorbate)
    size_lst: list = [[1,1,1]], 
    ads_slab_image = None, #need to be not None when placement_mode is most_stable_sties
    adsorbate_xy_lst = None,
    save_to_db: bool = True,
    visualize: bool = True,
):
    def plot_slab(ase_atoms, fig_name):
        """
        Plots and saves images of the slab from different perspectives.

        This function visualizes the slab with constraints highlighted and saves the plots to files.
        It plots the slab from three different perspectives and saves each image with a unique figure name based on
        the slab order.

        Args:
            ase_atoms (ASE Atoms object): The slab to visualize.
            fig_name (str): The base name for the output figure files, which will be appended with order indices.

        """
        from copy import deepcopy

        # hack a constraint visualization
        vis_atoms = deepcopy(ase_atoms)

        fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
        plot_atoms(vis_atoms, axarr[0], rotation=("270x,0y,0z"))
        axarr[0].set_title("0 degree rotation")

        try:
            index_lst = vis_atoms.constraints[0].get_indices()
        except:
            index_lst = []
        color_lst = ["green" if i != len(vis_atoms) - 1 else "yellow" for i in range(len(vis_atoms))]
        for i in index_lst:
            color_lst[i] = "red"
        plot_atoms(vis_atoms, axarr[1], rotation=("270x,0y,0z"), colors=color_lst)
        axarr[1].set_title(
            "0 dgree rotation\n(Green means free, red means fixed, yellow means adsorbate)"
        )
        pymatgen_slab = AseAtomsAdaptor.get_structure(vis_atoms)
        plot_slab_pymatgen(pymatgen_slab,axarr[2],repeat=1,window=0.8,adsorption_sites=False,scale=1,decay=0.2)
        axarr[2].set_title(
            "Top view"
        )
        fig.savefig(f"{fig_name}.png")

    def write_to_database(base_slab,adsorbate_lst,results,base_slab_facet,idx,size,placement_mode,save_to_db,visualize):
        size_str = '.'.join([str(i) for i in size])
        reduce_formula = base_slab.get_chemical_formula(empirical=True)
        base_slab_db_name = '.'.join([reduce_formula,base_slab_facet,'id',idx,placement_mode,'db'])
        base_slab_db_path = Path(base_slab_db_name)
        if save_to_db:
            with connect(base_slab_db_path) as base_db:
                base_db.write(
                    atoms=base_slab,
                    size=size_str,
                    )
        for adsorbate in adsorbate_lst:
            for ads_site, site_dict in results[adsorbate].items():
                for site_coord, site_dict in site_dict.items():
                    slab_atoms = site_dict['structure']
                    if save_to_db:
                        ads_slab_db_name = '.'.join([reduce_formula,base_slab_facet,'id',idx,'ads',str(adsorbate),placement_mode,'db'])
                        ads_slab_db_path = Path(ads_slab_db_name)
                        with connect(ads_slab_db_path) as ads_db:
                            ads_db.write(
                                atoms=slab_atoms,
                                ads_site=ads_site,
                                size=size_str,
                                )
                    if visualize:
                        plot_slab(slab_atoms,f"{str(base_slab.symbols)}.{base_slab_facet}.id.{idx}.{adsorbate}.{ads_site}.{site_coord}.{size_str}") 
    
    if placement_mode == 'all_sites':
        base_slab = get_atoms(**slab_image)*size_lst[0]
        results = adsorption.generate_adsorbed_structures(
            surface=base_slab,
            adsorbates=adsorbate_lst,
            use_all_sites=True,
            write_to_disk=False,
        )
        base_slab_facet = os.path.basename(slab_image['image_file']).split('.')[1]
        idx = str(slab_image['index'])
        write_to_database(base_slab,adsorbate_lst,results,base_slab_facet,idx,size_lst[0],placement_mode,save_to_db,visualize)
                    
    elif placement_mode == 'most_stable_site':
        if ads_slab_image is None:
            raise RuntimeError("`ads_slab_image` cannot be None, if placement_mode is 'most_stable_site'.")
        base_slab_id = int(ads_slab_image['image_file'].split('/')[-1].split('.id')[1].split('.')[1])
        slab_image['index'] = base_slab_id
        for size in size_lst:
            base_slab = get_atoms(**slab_image)*size
            ads_slab_lst = get_images(**ads_slab_image)
            ads_slab_energy = [ads_slab.get_potential_energy() for ads_slab in ads_slab_lst]
            ads_slab_energy_min_idx = np.argsort(ads_slab_energy)[0]

            ads_slab_most_stable = ads_slab_lst[ads_slab_energy_min_idx]
            ads_x, ads_y = ads_slab_most_stable.positions[-1,:-1]

            results = adsorption.generate_adsorbed_structures(
                surface=base_slab,
                adsorbates=adsorbate_lst,
                adsorption_sites = {"most_stable_site": [(ads_x, ads_y)]},
                write_to_disk=False,
            )
            base_slab_facet = os.path.basename(slab_image['image_file']).split('.')[1]
            idx = str(slab_image['index'])
            write_to_database(base_slab,adsorbate_lst,results,base_slab_facet,idx,size,placement_mode,save_to_db,visualize)
    elif placement_mode == "ads_convergence":
        base_slab_lst = get_images(**slab_image)
        for i,base_slab in enumerate(base_slab_lst):
            base_slab = base_slab*size_lst[0]
            ads_slab_lst = get_images(**ads_slab_image)
            if ads_slab_image is None and adsorbate_xy_lst is None:
                raise RuntimeError("`ads_slab_image` and `adsorbate_xy_lst` cannot be both None, if placement_mode is 'ads_convergence'.")
            elif ads_slab_image is not None: 
                ads_slab_energy = [ads_slab.get_potential_energy() for ads_slab in ads_slab_lst]
                ads_slab_energy_min_idx = np.argsort(ads_slab_energy)[0]

                ads_slab_most_stable = ads_slab_lst[ads_slab_energy_min_idx]
                ads_x, ads_y = ads_slab_most_stable.positions[-1,:-1]
            elif adsorbate_xy_lst is not None:
                assert len(adsorbate_xy_lst) == len(base_slab_lst), "provided adsorbate_xy_lst length should be the same as the base_slab_lst length."
                ads_x, ads_y = adsorbate_xy_lst[i]

            results = adsorption.generate_adsorbed_structures(
                surface=base_slab,
                adsorbates=adsorbate_lst,
                adsorption_sites = {"most_stable_site": [(ads_x, ads_y)]},
                write_to_disk=False,
            )
            base_slab_facet = os.path.basename(slab_image['image_file']).split('.')[1]
            idx = os.path.basename(slab_image['image_file']).split('.')[-2]
            write_to_database(base_slab,adsorbate_lst,results,base_slab_facet,idx,size_lst[0],placement_mode,save_to_db,visualize)

    
    return {}

if __name__ == "__main__":
    import yaml

    with open(
        "/jet/home/jpu/projects/projects/asimtools/use_case/adsorption/sim_input_ads_sites.yaml"
    ) as stream:
        args = yaml.safe_load(stream)
    args = args["args"]
    generate_sites(**args)
