from autocat import adsorption
from ase.db import connect 
from pathlib import Path
import matplotlib.pyplot as plt 
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import plot_slab as plot_slab_pymatgen
from asimtools.utils import get_atoms

def generate_lowest_sites(
    slab_image,
    adsorbate_lst: list,  # one adsorbate e.g. Li
    save_to_db = True,
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
    base_slab = get_atoms(**slab_image)
    results = adsorption.generate_adsorbed_structures(
        surface=base_slab,
        adsorbates=adsorbate_lst,
        use_all_sites=True,
        write_to_disk=False,
    )
    for adsorbate in adsorbate_lst:
        for ads_site, site_dict in results[adsorbate].items():
            for site_coord, site_dict in site_dict.items():
                slab_atoms = site_dict['structure']
                if save_to_db:
                    ads_slab_db_name = '.'.join([slab_image['image_file'].split('/')[-1].split('.db')[0],'id',str(slab_image['index']),'ads','db'])
                    ads_slab_db_path = Path(ads_slab_db_name)
                    ads_db = connect(ads_slab_db_path)
                    ads_db.write(
                        atoms=slab_atoms,
                        ads_site=ads_site,
                        adsorbate=adsorbate
                        )
                if visualize:
                    plot_slab(slab_atoms,f"id.{slab_image['index']}.{adsorbate}.{ads_site}.{site_coord}") 
        # print(type(results['Li']['ontop']['1.107_1.74']['structure']))
    return {}

if __name__ == "__main__":
    import yaml

    with open(
        "/jet/home/jpu/projects/projects/asimtools/use_case/adsorption/sim_input_ads_sites.yaml"
    ) as stream:
        args = yaml.safe_load(stream)
    args = args["args"]
    generate_sites(**args)
