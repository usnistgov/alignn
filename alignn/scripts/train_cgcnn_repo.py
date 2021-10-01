"""Module to train CGCNN model from repo."""
import os
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import time
import warnings

warnings.filterwarnings("ignore")

# id_tag='jid'
# dataset_name='dft_2d'
# dataset=data(dataset_name)
# prop='optb88vdw_bandgap'
# formation_energy_peratom


def cgcnn_pred(id_tag="jid", dataset_name="dft_2d", prop="optb88vdw_bandgap"):
    """Train CGCNN model from repo."""
    t1 = time.time()
    dataset = data(dataset_name)
    cgcnn_folder = os.path.join(os.getcwd(), "cgcnn")
    if not os.path.exists(cgcnn_folder):
        cmd = "git clone https://github.com/txie-93/cgcnn.git"
        os.system(cmd)
    cwd = os.getcwd()
    os.chdir(cgcnn_folder)
    local_name = "data_dir_" + dataset_name + "_" + prop
    new_dir = os.path.join(os.getcwd(), local_name)

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    os.chdir(new_dir)
    cmd = (
        "wget https://raw.githubusercontent.com/txie-93/cgcnn"
        + "/master/data/sample-regression/atom_init.json"
    )
    if not os.path.exists("atom_init.json"):
        os.system(cmd)

    f = open("id_prop.csv", "w")
    for i in dataset:
        if i[prop] != "na":
            line = i[id_tag] + "," + str(i[prop]) + "\n"
            f.write(line)
            atoms = Atoms.from_dict(i["atoms"])
            name = i[id_tag] + ".cif"
            atoms.write_cif(name)

    f.close()
    os.chdir(cgcnn_folder)
    cmd = (
        "python main.py --train-ratio .8 --val-ratio .1 --test-ratio .1 "
        + "--batch-size 256 --lr 0.01 --workers 2 --epoch 300 "
        + local_name
    )
    os.system(cmd)
    os.chdir(cwd)
    t2 = time.time()
    print("Time:", t2 - t1)


if __name__ == "__main__":
    # cgcnn_pred(prop="formation_energy_peratom",dataset_name='dft_3d',id_tag='jid')
    cgcnn_pred(
        prop="optb88vdw_total_energy", dataset_name="dft_3d", id_tag="jid"
    )
