"""Module to download and load pre-trained ALIGNN models."""
import requests
import os
import zipfile
from tqdm import tqdm
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import tempfile
import torch
import sys
from jarvis.db.jsonutils import loadjson
import argparse
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph

all_models = loadjson(
    str(os.path.join(os.path.dirname(__file__), "pretrained_models.json"))
)


parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network Pretrained Models"
)
parser.add_argument(
    "--model_name",
    default="jv_formation_energy_peratom_alignn",
    help="Choose a model from these "
    + str(len(list(all_models.keys())))
    + " models:"
    + ", ".join(list(all_models.keys())),
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--file_path",
    default="alignn/examples/sample_data/POSCAR-JVASP-10.vasp",
    help="Path to file.",
)

parser.add_argument(
    "--cutoff",
    default=8,
    help="Distance cut-off for graph constuction"
    + ", usually 8 for solids and 5 for molecules.",
)


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def get_prediction(
    model_name="jv_formation_energy_peratom_alignn",
    atoms=None,
    cutoff=8,
):
    """Get model with progress bar."""
    tmp = all_models[model_name]
    url = tmp[0]
    output_features = tmp[1]
    zfile = model_name + ".zip"
    path = str(os.path.join(os.path.dirname(__file__), zfile))
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    zp = zipfile.ZipFile(path)
    names = zp.namelist()
    for i in names:
        if "checkpoint_" in i and "pt" in i:
            tmp = i
            # print("chk", i)
    # print("Loading the zipfile...", zipfile.ZipFile(path).namelist())
    data = zipfile.ZipFile(path).read(tmp)
    model = ALIGNN(
        ALIGNNConfig(name="alignn", output_features=output_features)
    )
    new_file, filename = tempfile.mkstemp()
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    if os.path.exists(filename):
        os.remove(filename)

    # print("Loading completed.")
    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(cutoff))
    out_data = (
        model([g.to(device), lg.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )
    return out_data


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    model_name = args.model_name
    file_path = args.file_path
    file_format = args.file_format
    cutoff = args.cutoff
    if file_format == "poscar":
        atoms = Atoms.from_poscar(file_path)
    elif file_format == "cif":
        atoms = Atoms.from_cif(file_path)
    elif file_format == "xyz":
        atoms = Atoms.from_xyz(file_path, box_size=500)
    elif file_format == "pdb":
        atoms = Atoms.from_pdb(file_path, max_lat=500)
    else:
        raise NotImplementedError("File format not implemented", file_format)

    out_data = get_prediction(
        model_name=model_name, cutoff=float(cutoff), atoms=atoms
    )

    print("Predicted value:", model_name, file_path, out_data)


# x = get_model()
# print(x)
