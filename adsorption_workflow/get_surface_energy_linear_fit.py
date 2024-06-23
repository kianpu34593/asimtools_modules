from asimtools.utils import get_atoms
import numpy as np
from glob import glob 

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.ticker

import itertools

import matplotlib.pyplot as plt

import os
def get_surface_energy_linear_fit(
    path_to_all_clean_slab_db,
    actual_layer_lst = None,
):

    def plot_precision_setup(ax,x_pres=1,y_pres=0.01,y_sig_fig=2):
        #f = lambda x,pos: str(np.round(x,decimals=2)).rstrip('0').rstrip('.')
        # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(x_pres))
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(y_pres))
        # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(f'%.{y_sig_fig}f'))
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

    path_to_all_clean_slab_db = glob(path_to_all_clean_slab_db)
    num_of_atoms_surface_area_slab_energy_tuple_lst=[]
    num_of_layers_lst = []
    for path_to_db in path_to_all_clean_slab_db:
        basename_lst = os.path.basename(path_to_db).split('.')
        if 'ads' in basename_lst:
            continue
        clean_slab = get_atoms(path_to_db)
        num_of_layers = len(np.unique(detect_cluster(clean_slab)[1]))
        if actual_layer_lst is not None:
            if num_of_layers in actual_layer_lst:
                num_of_atoms_surface_area_slab_energy_tuple_lst.append((len(clean_slab),2*clean_slab.cell[0][0]*clean_slab.cell[1][1],clean_slab.get_potential_energy()))
                num_of_layers_lst.append(num_of_layers)
            else:
                continue
        else:
            num_of_atoms_surface_area_slab_energy_tuple_lst.append((len(clean_slab),2*clean_slab.cell[0][0]*clean_slab.cell[1][1],clean_slab.get_potential_energy()))
            num_of_layers_lst.append(num_of_layers)
    sorted_lst = sorted(num_of_atoms_surface_area_slab_energy_tuple_lst,key=lambda x: x[0])
    num_of_atoms_lst =  np.array(list(zip(*sorted_lst))[0])
    surface_area_lst = np.array(list(zip(*sorted_lst))[1])
    slab_energy_lst = np.array(list(zip(*sorted_lst))[2])
    num_of_layers_lst = np.sort(num_of_layers_lst)

    fitted_bulk_potential_energy=np.round(np.polyfit(num_of_atoms_lst,slab_energy_lst,1)[0],decimals=5)
    surf_energy_lst=(1/surface_area_lst)*(slab_energy_lst-num_of_atoms_lst*fitted_bulk_potential_energy)

    fig, ax = plt.subplots(ncols=1,nrows=1)
    plot_precision_setup(ax)
    ax.plot(num_of_layers_lst,surf_energy_lst,'-o')
    ax.set_xticks(num_of_layers_lst) 
    ax.set_xticklabels([str(i) for i in num_of_layers_lst])
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel("Surface Energy (eV/A^2)")
    ax.set_ylim([np.min(surf_energy_lst)-0.01,np.max(surf_energy_lst)+0.01])

    ax.grid()
    plt.tight_layout()
    plt.savefig("surface_energy_convergence.png")

    return {"layer": [float(i) for i in num_of_layers_lst],
            "surface energy (eV/A^2)": [float(i) for i in surf_energy_lst],
            }