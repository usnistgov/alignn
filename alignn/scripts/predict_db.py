"""Module to predict for a DB of Atoms."""
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn import ALIGNN
# from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.figshare import data


model_path = "JV15/jv_optb88vdw_bandgap_alignn/checkpoint_300.pt"
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = ALIGNN()
model.load_state_dict(torch.load(model_path, map_location=device)["model"])
# model=torch.load('checkpoint_250.pt')['model']
model.to(device)
model.eval()


def predict_for_db(
    name="polymer_genome",
    prop="gga_gap",
    filename="predictions.csv",
    id_tag="id",
):
    """Predict using ALIGNN for a DB."""
    db = data(name)
    filename = name + "_" + prop + "_v1_" + filename
    f = open(filename, "w")
    line = "id,original,out_data,num_atoms,formula,spacegroup_number\n"
    f.write(line)
    for i in db:
        src = i["source_folder"]
        if "vol" not in src:
            atoms = Atoms.from_dict(i["atoms"])
            id = i[id_tag]

            g, lg = Graph.atom_dgl_multigraph(atoms)
            out_data = (
                model([g.to(device), lg.to(device)])
                .detach()
                .cpu()
                .numpy()
                .flatten()
                .tolist()[0]
            )
            original = i[prop]
            line = (
                str(id)
                + ","
                + str(original)
                + ","
                + str(out_data)
                + ","
                + str(atoms.num_atoms)
                + ","
                + str(i["formula"])
                + ","
                + str(i["spacegroup_number"])
                + str("\n")
            )
            f.write(line)
            # print (line)
    f.close()


predict_for_db(name="qe_tb", prop="indir_gap", id_tag="jid")
"""
import pandas as pd
df=pd.read_csv('qe_tb_indir_gap_predictions.csv')
from sklearn.metrics import mean_absolute_error
original=df['original'].values
out_data=df['out_data'].values
mae=mean_absolute_error(original,out_data)
df['error']=abs(df['original']-df['out_data'])
tol=0.5
df2=(df[df['error']>tol])
print (len(df2))
"""
