from asimtools.utils import get_atoms, get_images
import numpy as np

def get_ads_xy_positions(
    ads_slab_image=None,  
):
    ads_slab_lst = get_images(**ads_slab_image)

    ads_slab_energy = [
        ads_slab.get_potential_energy() for ads_slab in ads_slab_lst
    ]
    ads_slab_energy_min_idx = np.argsort(ads_slab_energy)[0]

    ads_slab_most_stable = ads_slab_lst[ads_slab_energy_min_idx]
    ads_x, ads_y = ads_slab_most_stable.positions[-1, :-1]

    return {"positions":[float(np.round(ads_x,decimals=2)), float(np.round(ads_y,decimals=2))]}