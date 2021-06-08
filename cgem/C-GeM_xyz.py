import sys
from iodata import load_one
from cgem import CGem, get_protein_parameters
from cgem.attypes import classify_atom
from cgem.attypes import classify_charge
from cgem.utils   import write_xyz
from cgem.utils   import load_xyz

# load protein
if(len(sys.argv)!=2):
    print("Please input:\n1.pdb file")
    sys.exit()


input_file  = (sys.argv[1]) #input Trajectory file 
#pdb = load_one('1a4w_lig_convert.pdb')
#pdb = load_one(input_file)
numbers,coords_c = load_xyz(input_file)
# get latest CGem protein parameters
atoms = classify_atom(numbers,coords_c,[1,6,7])
atoms, coords_s = classify_charge(atoms,coords_c,1) #third entry is the net charge of the protein
data = get_protein_parameters()
shells=['X']*len(coords_s)

# make CGem model
#model = CGem.from_molecule(pdb.atnums, pdb.atcoords, coords_s=None, opt_shells=True, min_dist_s=1.0e-5, **data)
model = CGem.from_molecule(numbers,coords_c, coords_s=coords_s, opt_shells=False, min_dist_s=1.0e-5, **data)
print(atoms,model.charge_c)
#write_xyz(atoms,pdb.atcoords,"cores.xyz")
#write_xyz(shells,coords_c,"shells.xyz")
#print(model.coords_s)   # optimized shell coordinates
#print(model.energy)     # energy                           
