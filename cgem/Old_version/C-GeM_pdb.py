from iodata import load_one
from cgem import CGem, get_protein_parameters
# load protein
#pdb = load_one('1a4w_lig_convert.pdb')
pdb = load_one('test.pdb')
# get latest CGem protein parameters
data = get_protein_parameters()
print(data)
# make CGem model
model = CGem.from_molecule(pdb.atnums, pdb.atcoords, coords_s=None, opt_shells=True, min_dist_s=1.0e-5, **data)
print(model.coords_s)   # optimized shell coordinates
print(model.energy)     # energy                           
