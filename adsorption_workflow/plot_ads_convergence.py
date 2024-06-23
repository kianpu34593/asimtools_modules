from ase.db import connect
import numpy as np
from typing import Union
from glob import glob 

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage

import itertools

import matplotlib.pyplot as plt

def plot_ads_convergence(
    path_to_all_ads_slab_db,
    path_to_clean_slab_db,
    ):
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
    def get_ads_energy(
        path_to_db,
        ):
        ads_slab_db = connect(path_to_db)

        ads_energy_per_site_lst = [
            atomsrow.ads_energy_per_site for atomsrow in ads_slab_db.select()
        ]
        min_ads_energy_per_site = np.sort(ads_energy_per_site_lst)[0]

        num_of_layers = len(np.unique(detect_cluster(ads_slab_db.get_atoms(id=1))[1]))
        return num_of_layers, min_ads_energy_per_site
    path_to_all_ads_slab_db = glob(path_to_all_ads_slab_db)
    path_to_clean_slab_db = glob(path_to_clean_slab_db)
    layers_ads_energy_tuple_lst = []
    for path_to_db in path_to_all_ads_slab_db:
        results = get_ads_energy(path_to_db)
        layers_ads_energy_tuple_lst.append(results)
    layers_ads_energy_tuple_lst_sorted = sorted(layers_ads_energy_tuple_lst,key=lambda x: x[0])
    ads_energy_lst = [i[1] for i in layers_ads_energy_tuple_lst_sorted]
    with connect(path_to_clean_slab_db[0]) as my_db:
        layer_lst = [atomsrow.actual_layer for atomsrow in my_db.select()]
    plt.plot(layer_lst, ads_energy_lst,"-o")
    plt.xticks(layer_lst, layer_lst) 
    plt.xlabel("No. of layers")
    plt.ylabel("Ads energy (eV/site)")
    plt.savefig("ads_convergence.png")
    return {"layer": [float(i) for i in layer_lst], "ads_energy (eV/site)": [float(i) for i in ads_energy_lst]}

    
