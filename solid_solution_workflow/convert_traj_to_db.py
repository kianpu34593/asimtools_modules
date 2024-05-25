from glob import glob
import os
from ase.io import read
from tqdm import tqdm
from ase.db import connect
import re
import numpy as np
def convert_traj_to_db(traj_dir_path,orig_element,subs_element,path_to_db):
    files_lst = glob(traj_dir_path+'/*/*.traj')
    dir_path = os.path.dirname(files_lst[0])
    atoms_lst = []#[1 for _ in range(len(files_lst))]
    for i,fi in enumerate(tqdm(files_lst)):
        atoms = read(fi)
        symbols = atoms.get_chemical_symbols()
        unique_symbols = np.unique(symbols)
        atoms_count_dict = {i: np.sum(np.array(symbols) == i) for i in unique_symbols}
        TM_total = atoms_count_dict[orig_element] + atoms_count_dict[subs_element]
        subs_TM_count = atoms_count_dict[subs_element]
        subs_element_conc = float(np.round(int(subs_TM_count) / int(TM_total), decimals=4))
        atoms_lst.append((atoms,subs_element_conc))
    with connect(path_to_db) as my_db:
        for atoms,subs_conc in tqdm(atoms_lst):
            my_db.write(atoms=atoms,subs_conc=subs_conc)
    return {}