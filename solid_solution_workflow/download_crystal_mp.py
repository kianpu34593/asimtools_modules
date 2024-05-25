from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter
api_key='mui9dtio4ndhyHOSaVmyCHa5Vs6wXVPP'
a = MPRester(api_key)

results = a.materials.search(material_ids=[f"mp-29210"], fields=["structure","formula_pretty"])[0]
initial_structures = results.structure#.make_supercell([2, 2, 2]) #only use this when the number of atoms is less than 2
# print(results.as_dict())
formula = results.formula_pretty
CifWriter(initial_structures, symprec=0.1, refine_struct=True).write_file(
    "/jet/home/jpu/projects/projects/asimtools/use_case/slab_creation/Li2Ga.cif"
)