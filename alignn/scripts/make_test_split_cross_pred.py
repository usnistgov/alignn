from jarvis.db.jsonutils import loadjson
from alignn.models.alignn_layernorm import ALIGNN  # , ALIGNNConfig

# from alignn.models.alignn import ALIGNN
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.figshare import data
from sklearn.metrics import mean_absolute_error
import numpy as np
from jarvis.core.utils import chunks

path = "ids_train_val_test.json"
item = "id_test"
dataset = "dft_3d"
prop = "optb88vdw_bandgap"
# "formation_energy_peratom"
model_path = "checkpoint_300.pt"
id_tag = "jid"

d = loadjson(path)
test_ids = d[item]

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
    name="dft_3d", prop="formation_energy_peratom", id_tag="jid", ids=[]
):
    """Predict using ALIGNN for a DB."""
    db = data(name)
    df = pd.DataFrame(db)
    x = []
    y = []
    for i in ids:
        atoms = Atoms.from_dict(df[df.jid == i]["atoms"].values[0])
        g, lg = Graph.atom_dgl_multigraph(atoms)
        out_data = (
            model([g.to(device), lg.to(device)])
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist()[0]
        )
        original = df[df.jid == i][prop].values[0]
        x.append(original)
        y.append(out_data)
    x = np.array(x)
    y = np.array(y)
    mae = mean_absolute_error(x, y)
    return mae


n_splits = 5
n_size = int(len(test_ids) / n_splits)
chnks = chunks(test_ids, n_size)
print(len(test_ids), len(chnks), n_size)
maes = []
for i in chnks:
    mae = predict_for_db(ids=i, prop=prop)
    maes.append(mae)
    print("MAE", mae)
maes = np.array(maes)
print(maes.mean(), maes.std())
