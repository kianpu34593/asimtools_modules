from ase.db import connect
import os 
from tqdm import tqdm

def divide_database(path_to_db,limit=999):
    my_db = connect(path_to_db)
    db_name = os.path.basename(path_to_db)
    i = 0
    if len(my_db)>limit:
        atoms_lst = []
        key_value_pairs_lst = []
        for atoms_row in tqdm(my_db.select(),total=len(my_db)):
            atoms_lst.append(atoms_row.toatoms())
            key_value_pairs_lst.append(atoms_row.key_value_pairs)
            if len(atoms_lst)>=limit:
                new_db_name='.'.join([db_name.split('.db')[0], f'{i}', 'db'])
                with connect(new_db_name) as my_new_db:
                    for atoms,key_value_pairs in zip(atoms_lst,key_value_pairs_lst):
                        my_new_db.write(atoms,**key_value_pairs)
                atoms_lst = []
                key_value_pairs_lst = []
                i+=1
        if len(atoms_lst)!=0:
            new_db_name='.'.join([db_name.split('.db')[0], f'{i}', 'db'])
            with connect(new_db_name) as my_new_db:
                for atoms,key_value_pairs in zip(atoms_lst,key_value_pairs_lst):
                    my_new_db.write(atoms,**key_value_pairs)
            atoms_lst = []
            key_value_pairs_lst = []
    return {}

if __name__ == "__main__":
    path_to_db = "/scratch/venkvis_root/venkvis/kianpu/projects/afx/solid_solution/NiF3_FeF3/NiF3_subs_Fe/NiF3_222_subs_Fe.db"
    divide_database(path_to_db)