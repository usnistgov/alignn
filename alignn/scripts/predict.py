"""Module to predict using a trained model."""
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn import ALIGNN
from jarvis.analysis.structure.spacegroup import Spacegroup3D

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
model = ALIGNN()
model.load_state_dict(torch.load("checkpoint_250.pt")["model"])
# model=torch.load('checkpoint_250.pt')['model']
model.to(device)
model.eval()
atoms = Atoms.from_poscar("POSCAR")
cvn = Spacegroup3D(atoms).conventional_standard_structure

g, lg = Graph.atom_dgl_multigraph(atoms)
out_data = (
    model([g.to(device), lg.to(device)])
    .detach()
    .cpu()
    .numpy()
    .flatten()
    .tolist()[0]
)
print("original", out_data)

g, lg = Graph.atom_dgl_multigraph(cvn)
out_data = (
    model([g.to(device), lg.to(device)])
    .detach()
    .cpu()
    .numpy()
    .flatten()
    .tolist()[0]
)
print("cvn", out_data)


atoms = atoms.make_supercell([3, 3, 3])
g, lg = Graph.atom_dgl_multigraph(atoms)
out_data = (
    model([g.to(device), lg.to(device)])
    .detach()
    .cpu()
    .numpy()
    .flatten()
    .tolist()[0]
)
print("supercell", out_data)
