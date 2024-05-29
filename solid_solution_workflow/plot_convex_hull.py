from ase.db import connect 
from asimtools.utils import get_atoms
import numpy as np
import matplotlib.pyplot as plt
def plot_convex_hull(path_to_db,
                        left_end_point_image,
                        right_end_point_image,
                        subs_element,
                        convex_point_lst:list=None):


    atoms_left = get_atoms(**left_end_point_image)
    atoms_right = get_atoms(**right_end_point_image)
    atoms_left_energy_per_atom = atoms_left.get_potential_energy()/len(atoms_left)
    atoms_right_energy_per_atom =atoms_right.get_potential_energy()/len(atoms_right)

    ss_db = connect(path_to_db)
    all_conc_lst = np.sort([atomsrow['subs_conc'] for atomsrow in ss_db.select()])
    unique_conc_lst = list(np.unique(all_conc_lst))
    all_conc_lst = [0]+list(all_conc_lst)+[1]
    all_energy_lst = [0 for _ in range(len(ss_db))]
    minmum_energy_lst = []
    minmum_xy_lst = []
    idx = 0

    for conc in unique_conc_lst:
        energy_lst_per_conc = []
        for atomsrow in ss_db.select(subs_conc=conc):
            try:
                atomsrow.energy
            except:
                continue
            energy_per_atom = atomsrow.energy/atomsrow.natoms
            adjusted_energy_per_atom = energy_per_atom - conc*atoms_right_energy_per_atom - (1-conc)*atoms_left_energy_per_atom
            energy_lst_per_conc.append((adjusted_energy_per_atom,atomsrow.id))
            all_energy_lst[idx]=adjusted_energy_per_atom
            idx+=1
        
        energy_lst_per_conc=sorted(energy_lst_per_conc, key = lambda x: x[0])
        minmum_energy_lst.append(energy_lst_per_conc[0][0])
        minmum_xy_lst.append((conc,energy_lst_per_conc[0][1]))
    
    fig = plt.figure() 
    plt.scatter(all_conc_lst,[0]+all_energy_lst+[0])
    unique_conc_lst = [0]+unique_conc_lst+[1]
    minmum_energy_lst = [0]+list(minmum_energy_lst)+[0]
    plt.plot(unique_conc_lst,minmum_energy_lst,'--')
    if convex_point_lst is not None:
        manual_conc_lst = np.array(unique_conc_lst)[convex_point_lst]
        manual_energy_lst = np.array(minmum_energy_lst)[convex_point_lst]
        plt.plot(manual_conc_lst,manual_energy_lst,'r',label='Convex Hull')
        plt.legend()
    plt.xlabel(f'{subs_element} Concentration')
    plt.ylabel('Formation energy (eV/atom)')

    current_positions = plt.gca().get_xticks()
    current_labels = [item.get_text() for item in plt.gca().get_xticklabels()]

    current_labels[0] = atoms_left.get_chemical_formula(empirical=True)
    current_labels[-1] = atoms_right.get_chemical_formula(empirical=True)
    plt.xticks(current_positions, current_labels)

    plt.savefig("Convex_hull.png")
    # minimum_dict={"Conc.":[i[0] for i in minmum_xy_lst],"Index":[i[1] for i in minmum_xy_lst]}
    # df = pd.DataFrame.from_dict(minimum_dict)
    # print(df)
    return {}