from alignn.models.alignn import ALIGNN, ALIGNNConfig
import torch
output_features =  1
#directory of checkpoint_file, basically, your optimized model
filename = 'DataSet_A_Model/checkpoint_150.pt'
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
model = ALIGNN(ALIGNNConfig(name="alignn", output_features=output_features))
model.load_state_dict(torch.load(filename, map_location=device)["model"])
model.eval()

import os
import csv
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
import re

cutoff = 8.0
max_neighbors = 12

#directory where you have all the poscar/cif and you would like to apply your optimized model on them.

sample_data_folder = '/Users/habibur/Habibur_Python_Scripts/alignn/alignn/data/'

# id_prop.csv; a csv file where you have all the ids of the poscar/cif in the first column and corresponding properties in the second column.

csv_file = 'id_prop.csv'
# In this output.csv file, all the ids and corresponding properties will be printed out.
output_file = 'output.csv'

with open(os.path.join(sample_data_folder, csv_file), newline='') as f:
    reader = csv.reader(f)
    file_list = [row[0] for row in reader]

atoms_list = []
for file in file_list:
    atoms = Atoms.from_cif(os.path.join(sample_data_folder, file))
    atoms_list.append(atoms)

g_list = []
lg_list = []
for atoms in atoms_list:
    g, lg = Graph.atom_dgl_multigraph(
        atoms, cutoff=float(cutoff), max_neighbors=max_neighbors
    )
    g_list.append(g)
    lg_list.append(lg)

out_data_list = []
for g, lg in zip(g_list, lg_list):
    out_data = (
        model([g.to(device), lg.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )
    out_data_str = str(out_data)
    # Extract data within square brackets
    match = re.search(r'\[(.*)\]', out_data_str)
    if match:
        out_data_list.append(match.group(1))
    else:
        out_data_list.append('')

with open(os.path.join(sample_data_folder, output_file), mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Filename', 'Output'])
    for i, file in enumerate(file_list):
        writer.writerow([file, out_data_list[i]])
