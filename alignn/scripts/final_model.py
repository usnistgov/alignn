from alignn.data import load_dataset
# from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
from alignn.train import train_dgl
from alignn.models.alignn import ALIGNN
import torch

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


dataset_name = "dft_3d"
k = "optb88vdw_total_energy"
id_tag = "jid"
model_path = (
    "ALL_DATASETS/JV15/jv_optb88vdw_total_energy_alignn/checkpoint_300.pt"
)
config_file = "ALL_DATASETS/JV15/jv_optb88vdw_total_energy_alignn/config.json"
config = loadjson(config_file)


dataset = load_dataset(name=dataset_name, target=k)
size = len(dataset)
batch_size = 64
config["n_train"] = size - 2 * batch_size
config["n_val"] = batch_size
config["n_test"] = batch_size
config["epochs"] = 300
config["batch_size"] = batch_size
config["dataset"] = dataset_name


tconf = TrainingConfig(**config)
print(tconf)
model = ALIGNN()
model.load_state_dict(torch.load(model_path, map_location=device)["model"])
model.to(device)

train_dgl(tconf, model)
"""
import sys
sys.exit()
train_loader, val_loader,  test_loader, prepare_bacth = get_train_val_loaders(
    dataset=dataset_name,
    target=k,
    n_train=size - 2,
    n_test=1,
    n_val=1,
    workers=4,
    id_tag=id_tag,
    batch_size=1,
)
#model.train()

"""
