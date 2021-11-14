"""Module to download and load pre-trained ALIGNN models."""
import requests
import os
import zipfile
from tqdm import tqdm
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import tempfile
import torch

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def get_model(
    url="https://figshare.com/ndownloader/files/31458679",
    output_features=1,
    tag="jv_formation_energy_peratom_alignn",
):
    """Get model with progress bar."""
    zfile = tag + ".zip"
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
            print("chk", i)
    print("Loading the zipfile...", zipfile.ZipFile(path).namelist())
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

    print("Loading completed.")
    return model


x = get_model()
print(x)
