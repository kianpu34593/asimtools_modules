from glob import glob
import os
from ase.io import read
from tqdm import tqdm
from ase.db import connect
import re
import numpy as np
def convert_chgnet_to_db(traj_dir_path,
                    orig_element,
                    subs_element):
    def files_sort_read_save(files_lst,orig_element,subs_element):
        files_id_lst = [int(f.split('/')[-2].split('__')[1]) for f in files_lst]
        files_lst_sorted = np.array(files_lst)[np.argsort(files_id_lst)]
        atoms_lst = []#[1 for _ in range(len(files_lst))]
        for i,fi in enumerate(files_lst_sorted):
            try:
                atoms = read(fi)
            except:
                basename = os.path.basename(fi)
                fi = fi.replace(basename,'image_input.xyz')
                atoms = read(fi)
            symbols = atoms.get_chemical_symbols()
            unique_symbols = np.unique(symbols)
            atoms_count_dict = {i: np.sum(np.array(symbols) == i) for i in unique_symbols}
            TM_total = atoms_count_dict[orig_element] + atoms_count_dict[subs_element]
            subs_TM_count = atoms_count_dict[subs_element]
            subs_element_conc = float(np.round(int(subs_TM_count) / int(TM_total), decimals=4))
            atoms_lst.append((atoms,subs_element_conc))
        basename = os.path.basename(files_lst[0]).split('.')[0]
        with connect(f"{basename}.db") as my_db:
            for atoms,subs_conc in atoms_lst:
                my_db.write(atoms=atoms,subs_conc=subs_conc)

    files_prediction_lst = glob(traj_dir_path+'/*/image_output.traj')
    files_sort_read_save(files_prediction_lst,orig_element,subs_element)
    
    files_relax_lst = glob(traj_dir_path+'/*/chgnet_relax.traj')
    if len(files_relax_lst) != 0:
        files_sort_read_save(files_relax_lst,orig_element,subs_element)

    return {}