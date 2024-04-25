from pymatgen.core.surface import generate_all_slabs
import pandas as pd
from asimtools.utils import get_atoms
import numpy as np
from pathlib import Path
import os
from pymatgen.io.cif import CifWriter
from ase.io import read, write
import itertools
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage


def detect_facet(
    bulk_image,
    gen_all_slab_dict,
    # max_miller_index: int,
    # layer: int=6,
    # vacuum_layer: int=10,
    # symmetric: bool=True,
    # in_unit_planes: bool=True,
    # center_slab: bool=True,
):
    """
    detech all the facts of a given materials maximum miller index
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

    bulk_struct = get_atoms(**bulk_image)
    slabgenall = generate_all_slabs(
        bulk_struct,
        **gen_all_slab_dict,
        # max_miller_index,
        # layer,
        # vacuum_layer,
        # center_slab=center_slab,
        # symmetrize=symmetric,
        # in_unit_planes=in_unit_planes,
    )
    # print('Miller Index'+'\t'+'Num of Different Shift(s)'+'\t'+'Shifts')
    # slab_M=[]
    # slabgenall_sym=[]
    miller_index_lst = []
    unique_cluster_lst = []
    shifts_lst = []
    for slab in slabgenall:
        temp_path = Path("temp.cif")
        CifWriter(slab).write_file(temp_path)
        slab_ase = read(temp_path)
        act_layer = len(np.unique(detect_cluster(slab_ase)[1]))
        miller_index = slab.miller_index
        slab_shift = np.round(slab.shift, decimals=4)
        miller_index_lst.append(miller_index)
        shifts_lst.append(slab_shift)
        unique_cluster_lst.append(act_layer)

    slabs_info_dict = {
        "shift": shifts_lst,
        "miller_index": miller_index_lst,
        "unique_cluster": unique_cluster_lst,
    }
    slabs_info_df = pd.DataFrame(
        slabs_info_dict
    )  # .set_index(["miller_index", "shift"])
    slabs_info_df = (
        slabs_info_df.groupby(["miller_index"])
        .agg(
            count=("shift", "size"),  # Count the number of unique shift values
            shift_lst=("shift", lambda x: x),  # Collect all shifts into a list
            actual_layer_lst=("unique_cluster", lambda x: x),
        )
        .reset_index()
    )
    slabs_info_df["max_index_value"] = slabs_info_df["miller_index"].apply(max)

    slabs_info_df = slabs_info_df.sort_values(by="max_index_value", ascending=True)

    slabs_info_df = slabs_info_df.drop(columns=["max_index_value"]).reset_index(
        drop=True
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print(slabs_info_df)
    os.remove("temp.cif")
    return {}
    # slab_M_unique = Counter(chain(*slab_M))
    # for key in list(slab_M_unique.keys()):
    #     print(str(key)+'\t'+str(slab_M_unique[key])+'\t\t\t\t'+str([np.round(slab.shift,decimals=4) for slab in slabgenall_sym if slab.miller_index==key]))


if __name__ == "__main__":
    import yaml

    with open(
        "/jet/home/jpu/projects/projects/asimtools/use_case/slab/sim_input_detect_facet.yaml"
    ) as stream:
        args = yaml.safe_load(stream)
    args = args["args"]
    detect_facet(**args)
