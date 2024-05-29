
from ase.db import connect
import numpy as np
def organize_top_candidates_from_mlip_relax(path_to_mlip_db,
                                            path_to_unrelax_db,
                                            top_percentile:int,
                                            ):

    ss_db = connect(path_to_mlip_db)
    all_conc_lst = np.sort([atomsrow['subs_conc'] for atomsrow in ss_db.select()])
    unique_conc_lst = list(np.unique(all_conc_lst))


    top_percentile_energy_dict={}
    for conc in unique_conc_lst:
        energy_lst_per_conc = []
        for atomsrow in ss_db.select(subs_conc=conc):
            energy_per_atom = atomsrow.energy/atomsrow.natoms
            energy_lst_per_conc.append((energy_per_atom,atomsrow.id))
    
        energy_lst_per_conc=sorted(energy_lst_per_conc, key = lambda x: x[0])
        
        top_precentile_idx = int(np.ceil(len(energy_lst_per_conc)*(top_percentile/100)))

        if top_precentile_idx < 10:
            if len(energy_lst_per_conc) < 10:
                top_precentile_idx = len(energy_lst_per_conc)
            else:
                top_precentile_idx = 10

        top_percentile_energy_dict[conc] = [i[1] for i in energy_lst_per_conc[:top_precentile_idx]]
    ss_unrelax_db = connect(path_to_unrelax_db)
    with connect(f"top_{top_percentile}%.db") as my_new_db:
        for conc, idx in top_percentile_energy_dict.items():
            for i in idx:
                atoms = ss_unrelax_db.get(id=i)
                my_new_db.write(atoms=atoms,subs_conc=conc,orig_id=i)

    return {}
    

    