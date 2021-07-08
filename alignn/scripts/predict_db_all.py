"""Module to predict for all DB's form. enp and gap."""
import torch
# from jarvis.core.atoms import Atoms
# from jarvis.core.graphs import Graph
from alignn.models.alignn import ALIGNN
# from jarvis.analysis.structure.spacegroup import Spacegroup3D
# from jarvis.db.figshare import data
from alignn.data import load_dataset, get_train_val_loaders
from jarvis.db.jsonutils import loadjson
import numpy as np
import time

gap_model_path = "JV15/jv_optb88vdw_bandgap_alignn/checkpoint_300.pt"
form_model_path = "JV15/jv_formation_energy_peratom_alignn/checkpoint_300.pt"
dataset_props = loadjson("dataset_props.json")

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


for i, j in dataset_props.items():
    if "dft_" not in i:
        try:
            for k in j:
                print("property", k)
                if (
                    "gap" in k
                    or "form" in k
                    or k == "f_enp"
                    or k == "_oqmd_delta_e"
                ):
                    if "mbj" not in k:
                        t1 = time.time()
                        id_tag = i.split("-")[0]
                        dataset_name = i.split("-")[1]
                        dataset = load_dataset(name=dataset_name, target=k)
                        size = len(dataset)
                        data_loader, b, c, d = get_train_val_loaders(
                            dataset=dataset_name,
                            target=k,
                            n_train=size - 2,
                            n_test=1,
                            n_val=1,
                            workers=4,
                            id_tag=id_tag,
                            batch_size=1,
                        )

                        if "gap" in k:
                            model = ALIGNN()
                            model.load_state_dict(
                                torch.load(
                                    gap_model_path, map_location=device
                                )["model"]
                            )
                            model.to(device)
                            model.eval()
                        if "form" in k or k == "f_enp" or k == "_oqmd_delta_e":
                            model = ALIGNN()
                            model.load_state_dict(
                                torch.load(
                                    form_model_path, map_location=device
                                )["model"]
                            )
                            model.to(device)
                            model.eval()
                        filename = dataset_name + "_" + k + "predictions.csv"
                        f = open(filename, "w")
                        f.write("id,target,prediction,difference\n")
                        targets = []
                        predictions = []
                        with torch.no_grad():
                            ids = (
                                data_loader.dataset.ids
                            )  # [test_loader.dataset.indices]
                            for dat, id in zip(data_loader, ids):
                                g, lg, target = dat
                                out_data = model([g.to(device), lg.to(device)])
                                out_data = out_data.cpu().numpy().tolist()
                                target = (
                                    target.cpu().numpy().flatten().tolist()
                                )
                                if len(target) == 1:
                                    target = target[0]
                                diff = abs(target - out_data)
                                f.write(
                                    "%s, %6f, %6f, %6f\n"
                                    % (id, target, out_data, diff)
                                )
                                targets.append(target)
                                predictions.append(out_data)
                        f.close()
                        from sklearn.metrics import mean_absolute_error

                        print(
                            "Test MAE:",
                            mean_absolute_error(
                                np.array(targets), np.array(predictions)
                            ),
                        )
                        t2 = time.time()
                        print("size", i, k, size, dataset_name, t2 - t1)
                        del data_loader
        except Exception as exp:
            print(i, k, exp)
            pass
