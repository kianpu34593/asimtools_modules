import numpy as np
from ase.db import connect
from ase.units import kB
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
import random
from glob import glob
from math import comb
import pandas as pd
from tqdm import tqdm
import itertools
import os

def determine_boltzmann_factor(path_to_db_lib,temperature,tolerance=0,print_info=False):
    def BoltzmannDistPlot(degeneracies,boltzmann_weights,conc):
        fig = plt.figure() 
        plt.bar(list(degeneracies.keys()), boltzmann_weights,width=0.001)
        plt.xlabel('Energy Level')
        plt.ylabel('Boltzmann Weights')
        plt.title('Boltzmann Distribution')
        plt.savefig(f"Conc.{conc}.BoltzmannDistPlot.png")

    data_lst = []
    db_lib_path_lst = glob(f"{path_to_db_lib}/*/")
    db_lib_basename_lst = [os.path.basename(i[:-1]) for i in db_lib_path_lst]
    all_conc_lst = np.array([float(db_lib_basename.split('.conc.')[-1]) for db_lib_basename in db_lib_basename_lst])
    sort_idx = np.argsort(all_conc_lst)
    all_conc_lst = np.sort(all_conc_lst)
    db_lib_path_lst = np.array(db_lib_path_lst)[sort_idx]
    for conc,db_lib_path in tqdm(zip(all_conc_lst,db_lib_path_lst),total=len(db_lib_path_lst)):
        print("Conc: ", conc)
        all_db_path_lst = glob(f"{db_lib_path}/*.db")
        all_ads_energy_lst = [None for _ in all_db_path_lst]
        for i,db_path in enumerate(all_db_path_lst):
            with connect(db_path) as ads_db:
                db_ads_energy_lst = [None for _ in range(len(ads_db))]
                for j,atomsrow in enumerate(ads_db.select()):
                    try:
                        atomsrow.toatoms().get_potential_energy()
                    except:
                        continue
                    ads_energy = atomsrow.ads_energy_per_site
                    # if ads_energy > 0.095:
                    #     print(atomsrow.id)
                    db_ads_energy_lst[j]=ads_energy
            all_ads_energy_lst[i] = db_ads_energy_lst
        all_ads_energy_lst = list(itertools.chain(*all_ads_energy_lst))
        print(f"Total number of elements: {len(all_ads_energy_lst)}")
        # Count null elements using filter() and None
        null_count = all_ads_energy_lst.count(None)
        all_ads_energy_lst = list(filter(lambda item: item is not None, all_ads_energy_lst))
        print(f"Number of null elements: {null_count}")
        
        if tolerance == 0:
            # Determine degeneracies
            unique_energies, counts = np.unique(all_ads_energy_lst, return_counts=True)
            
        else:
            linked = linkage(all_ads_energy_lst[:, None], method='single')
            cluster_labels = fcluster(linked, t=tolerance, criterion='distance')
            # Determine unique clusters and their counts (degeneracies)
            unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
            unique_energies = np.array([np.mean(all_ads_energy_lst[cluster_labels == cluster]) for cluster in unique_clusters])
        
        degeneracies = dict(zip(unique_energies, counts))
        degeneracies = dict(sorted(degeneracies.items(), key=lambda x:x[0]))

        # Calculate Boltzmann weights
        kT = temperature * kB
        boltzmann_weights = np.exp(-np.array(list(degeneracies.keys())) / kT) * np.array(list(degeneracies.values()))
        boltzmann_weights /= np.sum(boltzmann_weights)  # Normalize weights
        if print_info == True:
            
            print("unique energies:", np.round(list(degeneracies.keys()),decimals=4))
            print("Count:", list(degeneracies.values()))
            print("Total combination: ", comb(16,int(conc*16)))
            print("Total count: ", np.sum(list(degeneracies.values())))
            print("Boltzmann Weights:", boltzmann_weights)
        BoltzmannDistPlot(degeneracies,boltzmann_weights,conc)

        data = pd.DataFrame({
            'conc': np.repeat([conc], len(list(degeneracies.keys()))),
            'unique ads_energies_per_sites(eV/ads_sites)': np.round(list(degeneracies.keys()),decimals=4),
            'Boltzmann probability': boltzmann_weights,
        })
        data_lst.append(data)
    data = pd.concat(data_lst,ignore_index=True)
    data.to_csv(f"AllUniqueEnergyDataWithProbability.csv")

    fig = plt.figure() 
    for conc in all_conc_lst:
        data_conc = data[data['conc'] == conc]
        X = data_conc['conc'].to_numpy()
        E = data_conc['unique ads_energies_per_sites(eV/ads_sites)'].to_numpy()
        weights = data_conc['Boltzmann probability'].to_numpy()
        sorted_indices = np.argsort(weights)
        X_sorted = X[sorted_indices]
        E_sorted = E[sorted_indices]
        weights_sorted = weights[sorted_indices]
        sc = plt.scatter(X_sorted, E_sorted, c=weights_sorted, vmin=0, vmax=1, s=75, cmap='Reds',alpha=1)
        #plt.scatter(X_sorted, E_sorted, facecolors='none', edgecolors='black', s=75,alpha=0.25)
    cbar=plt.colorbar(sc,location='right')
    cbar.ax.tick_params(labelsize=12)  # Adjust the label size as needed
    plt.xlabel("Monolayer Li Conc.")
    plt.ylabel(r'$\Delta$ h (eV)')
    plt.savefig(f"Enthalpy_of_adsorption.png")
    return {}


# def RainCloudsPlot(path_to_db_bank, conc_lst):
#     data_lst = []
#     for conc in tqdm(conc_lst):
#         db_path_lst = glob(f"{path_to_db_bank}/{conc}/*.db")
#         atomsrow_lst = []
#         for db_path in db_path_lst:
#             with connect(db_path) as ads_db:
#                 atomsrow_lst+=list(ads_db.select())
#         all_ads_energy_lst = []
#         for i,atomsrow in enumerate(atomsrow_lst):
#             try:
#                 atomsrow.toatoms().get_potential_energy()
#             except:
#                 continue
#             ads_energy = atomsrow.ads_energy_per_site
#             # if ads_energy > 0.095:
#             #     print(atomsrow.id)
#             all_ads_energy_lst.append(ads_energy)
#         all_ads_energy_lst = np.array(all_ads_energy_lst)
#         data = pd.DataFrame({
#             'conc': np.repeat([conc], len(all_ads_energy_lst)),
#             'ads_energies_per_sites(eV/ads_sites)': all_ads_energy_lst,
#         })
#         data_lst.append(data)
#     data = pd.concat(data_lst,ignore_index=True)
#     # Create the raincloud plot
#     plt.figure(figsize=(8, 6))
#     ax = pt.RainCloud(x='conc', y='ads_energies_per_sites(eV/ads_sites)', data=data, 
#                     palette='Set2', width_viol=0.6, width_box=0.2, 
#                     box_showfliers=False, move=0.2)
#     plt.title('Raincloud Plot')
#     plt.xlabel("Monolayer Li Conc.")
#     plt.ylabel("Adsorption Enthalpy (eV/ads_site)")
#     plt.savefig(f"{path_to_db_bank}/Rainclouds.png")
#     plt.show()
#     data.to_csv(f"{path_to_db_bank}/AllEnergyData.csv")


if __name__ == "__main__":
    path_to_db_lib = "/jet/home/jpu/projects/projects/anodefree/asimtools_calc/production/Cu_mp-30/100_ads_chained_workflow/step-4"
    T=300
    determine_boltzmann_factor(path_to_db_lib,T,tolerance=0)
