# /wrk/knc6/oc/oc2/ocp/data/val_cgcnn_test/pred2.py
# conda activate ocp-models
from jarvis.core.atoms import Atoms
from jarvis.core.specie import atomic_numbers_to_symbols

# from ocpmodels.datasets import SinglePointLmdbDataset
import os, torch
from ase.io import read

# from ocpmodels.preprocessing import AtomsToGraphs
# from ocpmodels.models import CGCNN
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from jarvis.core.atoms import Atoms

# from ocpmodels.datasets import data_list_collater
from tqdm import tqdm
from jarvis.db.figshare import get_request_data
import json, zipfile
import numpy as np
import pandas as pd
from jarvis.db.jsonutils import loadjson, dumpjson
import os
from jarvis.core.atoms import Atoms
from jarvis.core.atoms import Atoms
from jarvis.core.specie import atomic_numbers_to_symbols
import os, torch
from ase.io import read
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from jarvis.core.atoms import Atoms
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import os
import zipfile
from tqdm import tqdm
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.data import get_torch_dataset
from torch.utils.data import DataLoader

dat = get_request_data(js_tag="ocp_all.json",url="https://figshare.com/ndownloader/files/40974599")
df = pd.DataFrame(dat)
path = "../../../../../benchmarks/AI/SinglePropertyPrediction/ocp_all_relaxed_energy.json.zip"
js_tag = "ocp_all_relaxed_energy.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))
train_ids = np.array(list(id_data["train"].keys()))
val_ids = np.array(list(id_data["val"].keys()))
test_ids = np.array(list(id_data["test"].keys()))
train_df = df[df["id"].isin(train_ids)]
val_df = df[df["id"].isin(val_ids)]  # [:take_val]
test_df = df[df["id"].isin(test_ids)]

print(test_df)
# https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/is2re/10k/base.yml

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

model_path = "/wrk/knc6/Software/alignn_calc/jarvis_leaderboard/jarvis_leaderboard/contributions/alignn_model/OCP/all/tempall/checkpoint_45.pt"
config = "/wrk/knc6/Software/alignn_calc/jarvis_leaderboard/jarvis_leaderboard/contributions/alignn_model/OCP/all/tempall/config.json"
config_params = loadjson(config)

model = ALIGNN(ALIGNNConfig(name="alignn",alignn_layers=2))
# model = ALIGNN(ALIGNNConfig(name="alignn",output_features=1,config_params))
model.load_state_dict(torch.load(model_path, map_location=device)["model"])
model.to(device)
model.eval()
def get_multiple_predictions(
    model="",
    atoms_array=[],
    ids_array=[],
    cutoff=8,
    neighbor_strategy="k-nearest",
    max_neighbors=12,
    use_canonize=True,
    target="prop",
    atom_features="cgcnn",
    line_graph=True,
    workers=0,
    filename="pred_data.json",
    include_atoms=True,
    pin_memory=False,
    output_features=1,
    batch_size=1,
    model_name="jv_formation_energy_peratom_alignn",
    print_freq=100,
):
    """Use pretrained model on a number of structures."""
    # import glob
    # atoms_array=[]
    # for i in glob.glob("alignn/examples/sample_data/*.vasp"):
    #      atoms=Atoms.from_poscar(i)
    #      atoms_array.append(atoms)
    # get_multiple_predictions(atoms_array=atoms_array)

    mem = []
    for i, ii in enumerate(atoms_array):
        info = {}
        info["atoms"] = ii.to_dict()
        info["prop"] = -9999  # place-holder only
        info["jid"] = str(ids_array[i])
        mem.append(info)

    # Note cut-off is usually 8 for solids and 5 for molecules
    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=True,
            use_canonize=use_canonize,
        )

    test_data = get_torch_dataset(
        dataset=mem,
        target="prop",
        neighbor_strategy=neighbor_strategy,
        atom_features=atom_features,
        use_canonize=use_canonize,
        line_graph=line_graph,
    )

    collate_fn = test_data.collate_line_graph
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    results = []
    with torch.no_grad():
        ids = test_loader.dataset.ids
        for dat, id in zip(test_loader, ids):
            g, lg, target = dat
            out_data = model([g.to(device), lg.to(device)])
            out_data = out_data.cpu().numpy().tolist()
            target = target.cpu().numpy().flatten().tolist()
            info = {}
            info["id"] = id
            info["pred"] = out_data
            results.append(info)
            print_freq = int(print_freq)
            if len(results) % print_freq == 0:
                print(len(results))
    df1 = pd.DataFrame(mem)
    df2 = pd.DataFrame(results)
    df2["jid"] = df2["id"]
    df3 = pd.merge(df1, df2, on="jid")
    save = []
    for i, ii in df3.iterrows():
        info = {}
        info["id"] = ii["id"]
        info["atoms"] = ii["atoms"]
        info["pred"] = ii["pred"]
        save.append(info)

    dumpjson(data=save, filename=filename)


# f=open('AI-SinglePropertyPrediction-relaxed_energy-ocp100k-test-mae.csv','w')
# f.write('id,target,prediction\n')
# f.write('id,target,scaled_target,prediction\n')
# print('id,actual,scaled,pred')
atoms_array = []
ids_array = []
for ii, i in tqdm(test_df.iterrows()):
    fname = i["id"]
    ids_array.append(fname)
    atoms = (Atoms.from_dict(i["atoms"]))
    atoms_array.append(atoms)

    # actual=i['relaxed_energy']
    # relaxed_energy = (actual-target_mean)/target_std
    # scaled=relaxed_energy
    # data = a2g.convert(atoms).to(device)
    # batch = data_list_collater([data], otf_graph=False)
    # out = model(batch)
    # pred=(out[0].detach().cpu().numpy().flatten().tolist()[0])*target_std+target_mean
    # line=str(fname)+','+str(actual)+','+str(pred) #+'\n'
    # line=str(i.sid)+','+str(actual)+','+str(scaled)+','+str(pred) #+'\n'
    # f.write(line+'\n')
# f.close()
#atoms_array=atoms_array[0:10]
#ids_array=ids_array[0:10]
get_multiple_predictions(
    model=model, atoms_array=atoms_array, ids_array=ids_array
)
from jarvis.db.jsonutils import loadjson
d=loadjson('pred_data.json')
f=open('AI-SinglePropertyPrediction-relaxed_energy-ocp_all-test-mae.csv','w')
f.write('id,prediction\n')
for i in d:
 line=i['id']+','+str(i['pred'])+'\n'
 f.write(line)
f.close()

