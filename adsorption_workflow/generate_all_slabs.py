import itertools
from collections import Counter
from pathlib import Path
import os

from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure
from asimtools.utils import get_atoms
from pymatgen.io.cif import CifWriter

from ase.io import read, write
from ase.visualize.plot import plot_atoms
from ase.constraints import FixAtoms
from ase.db import connect
from ase import Atom
from ase.build.tools import sort

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage


def generate_all_slabs(
    bulk_image,
    # mp_id: Union[int,str]=None,
    # api_key: str=None,
    slabgen_args: dict,  # { "miller_index", "min_slab_size", "min_vacuum_size", "center_slab", "in_unit_planes"}
    symmetrize: bool,
    num_fix_layer: int = 2,
    visualize: bool = True,
    print_dataframe: bool = True,
    save_to_db: bool = True,
):
    """
    Generates slabs from a given bulk structure image and various slab generation parameters, and optionally visualizes
    and prints the resultant slab properties.

    This function performs several steps:
    1. It initializes a slab generator with the provided bulk structure and slab generation arguments.
    2. It generates slabs, optionally applying symmetrization.
    3. It processes each slab to determine slab properties such as angles, layers, and composition.
    4. Optionally, it visualizes each slab.
    5. Optionally, it prints a DataFrame summarizing the slab properties.
    6. If a database path is provided, it saves each slab to a database.

    Args:
        bulk_image (dict): Parameters necessary to generate the bulk structure using `get_atoms` function.
        slabgen_args (dict): Arguments required for initializing the SlabGenerator. This dictionary must include
                             keys like 'miller_index', 'min_slab_size', 'min_vacuum_size', 'center_slab', and 'in_unit_planes'.
        symmetrize (bool): Whether to symmetrize the slabs generated.
        visualize (bool, optional): If set to True, each slab will be visualized and saved as an image. Defaults to True.
        print_dataframe (bool, optional): If set to True, prints a DataFrame summarizing slab properties. Defaults to True.
        path_to_db (str, optional): Path to the database where slab data will be stored. If None, no data is stored. Defaults to None.

    Returns:
        dict: A dictionary containing lists of slab properties such as shifts, order indices, angles, layer counts,
              atom counts, compositions, and ASE atoms objects for each generated slab.

    Raises:
        RuntimeError: If a supported `fix_mode` is not provided during the fixing layer process.
    """

    def detect_cluster(slab, tol=0.3):
        """
        Detects clusters of atoms within a given slab based on the z-coordinate positions using hierarchical clustering.

        This function calculates the pairwise distance between atoms along the z-axis and applies hierarchical clustering
        to group atoms within a specified tolerance distance.

        Args:
            slab (ASE Atoms object): The slab for which clusters of atoms need to be detected.
            tol (float, optional): The tolerance for distance within which atoms are considered part of the same cluster. Defaults to 0.3.

        Returns:
            tuple: A tuple containing:
                - slab_c (numpy.ndarray): Sorted z-coordinate positions of atoms in the slab.
                - clusters (list): A list of cluster labels for each atom.
        """
        n = len(slab)
        dist_matrix = np.zeros((n, n))
        slab_c = np.sort(slab.get_positions()[:, 2])
        for i, j in itertools.combinations(list(range(n)), 2):
            if i != j:
                cdist = np.abs(slab_c[i] - slab_c[j])
                dist_matrix[i, j] = cdist
                dist_matrix[j, i] = cdist
        condensed_m = squareform(dist_matrix)
        z = linkage(condensed_m)
        clusters = fcluster(z, tol, criterion="distance")
        return slab_c, list(clusters)

    def fix_layer(slab, fix_layer, fix_mode):
        """
        Applies constraints to fix atoms in specified layers of the slab based on clustering.

        This function identifies and fixes atoms in either the bottom layers of the slab. It currently supports only
        fixing the bottom layer atoms based on the cluster information obtained from `detect_cluster`.

        Args:
            slab (ASE Atoms object): The slab to which the constraints will be applied.
            fix_layer (int): The number of layers from the specified mode (e.g., bottom) to fix.
            fix_mode (str): The mode of fixing layers, currently only supports 'bottom'.

        Returns:
            ASE Atoms object: The slab with constraints applied to specified layers.

        Raises:
            RuntimeError: If a non-supported fix_mode is provided.
        """

        slab_c, cluster = detect_cluster(slab)
        sort_unique_cluster_index = sorted(set(cluster), key=cluster.index)
        if fix_mode == "bottom":
            unique_cluster_index = sort_unique_cluster_index[: int(fix_layer)]
            fix_mask = np.logical_or.reduce(
                [cluster == value for value in unique_cluster_index]
            )
            # fix_mask=slab.positions[:,2]<(max_height_fix+0.05) #add 0.05 Ang to make sure all bottom fixed
        else:
            raise RuntimeError("Only bottom fix_mode supported.")
        slab.set_constraint(FixAtoms(mask=fix_mask))
        return slab

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
        import matplotlib.patches as patches

        # hack a constraint visualization
        vis_atoms = deepcopy(ase_atoms)

        fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
        plot_atoms(vis_atoms, axarr[0], rotation=("270x,0y,0z"))
        axarr[0].set_title("0 degree rotation")
        plot_atoms(vis_atoms, axarr[1], rotation=("270x,90y,0z"))
        axarr[1].set_title("90 degree rotation")

        try:
            index_lst = vis_atoms.constraints[0].get_indices()
        except:
            index_lst = []
        color_lst = ["green" for _ in range(len(vis_atoms))]
        for i in index_lst:
            color_lst[i] = "red"
        plot_atoms(vis_atoms, axarr[2], rotation=("270x,0y,0z"), colors=color_lst)
        axarr[2].set_title(
            "Constraint Map\n(Green means free, red means fixed)\n0 dgree rotation"
        )
        fig.savefig(f"{fig_name}.png")

    bulk_struct = get_atoms(**bulk_image)
    slabgen = SlabGenerator(
        initial_structure=bulk_struct, **slabgen_args
    )  # bulk_pymatgen, miller_index, layer, vacuum_layer, center_slab, in_unit_planes,
    slabs_symmetric = slabgen.get_slabs(symmetrize=symmetrize)
    (
        shift_ls,
        order_ls,
        slab_ase_ls,
        num_different_layers_ls,
        num_atom_ls,
        composition_ls,
    ) = ([], [], [], [], [], [])
    for i, slab in enumerate(slabs_symmetric):
        temp_path = Path("temp.cif")
        CifWriter(slab).write_file(temp_path)
        slab_ase = read(temp_path)
        L = slab_ase.cell.lengths()[2]
        slab_ase.cell[2] = [0, 0, L]  # TO-THINK: break the symmetry?
        slab_ase.wrap()
        slab_ase.center()
        slab_ase.pbc = [True, True, False]
        slab_ase = sort(slab_ase, tags=slab_ase.positions[:, 2])
        slab_ase = fix_layer(slab_ase, num_fix_layer, "bottom")
        slab_ase_ls.append(slab_ase)
        shift_ls.append(np.round(slab.shift, decimals=4))
        order_ls.append(i)
        unique_cluster = np.unique(detect_cluster(slab_ase)[1])
        num_different_layers_ls.append(len(unique_cluster))
        num_atom_ls.append(len(slab_ase))
        composition_dict = dict(Counter(slab_ase.get_chemical_symbols()))
        total_num_atoms = len(slab_ase)
        composition_ls.append(
            {
                key: float(np.round(values / total_num_atoms, decimals=4))
                for key, values in composition_dict.items()
            }
        )
        if visualize:
            plot_slab(slab_ase, f"order.{i}")

    slabs_info_dict = {
        "shift": shift_ls,
        "order": order_ls,
        "actual_layer": num_different_layers_ls,
        "num_of_atoms": num_atom_ls,
        "composition": composition_ls,
        "ase_atoms": slab_ase_ls,
    }
    slabs_info_df = pd.DataFrame(slabs_info_dict).set_index(["shift", "order"])
    if print_dataframe:
        pd.set_option("display.max_columns", None)
        print(slabs_info_df[["actual_layer", "num_of_atoms", "composition"]])
    os.remove(temp_path)
    if save_to_db:
        formula = slab_ase.get_chemical_formula(empirical=True)
        miller_index = "".join([str(i) for i in slabgen_args["miller_index"]])
        slab_db_name = ".".join([formula, miller_index, "db"])
        slab_db_path = Path(slab_db_name)
        atoms_db = connect(slab_db_path)
        for i in range(len(slab_ase_ls)):
            atoms_db.write(
                atoms=slabs_info_dict["ase_atoms"][i],
                shift=slabs_info_dict["shift"][i],
                order=slabs_info_dict["order"][i],
                actual_layer=slabs_info_dict["actual_layer"][i],
                composition=str(slabs_info_dict["composition"][i]),
            )

    return {}


if __name__ == "__main__":
    import yaml

    with open(
        "/jet/home/jpu/projects/projects/asimtools/use_case/slab/sim_input_detect_facet.yaml"
    ) as stream:
        args = yaml.safe_load(stream)
    args = args["args"]
    generate_sites(**args)
