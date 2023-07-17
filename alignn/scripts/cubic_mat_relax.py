from jarvis.core.atoms import Atoms

# import pandas as pd
from tqdm import tqdm

# from jarvis.core.graphs import Graph
from alignn.ff.ff import (
    # default_path,
    #    ev_curve,
    #    surface_energy,
    #    vacancy_formation,
    ForceField,
    #    fd_path,
    alignnff_fmult,
    #    get_interface_energy,
)
from jarvis.db.jsonutils import loadjson, dumpjson

# from jarvis.db.figshare import data

d = loadjson("data_1.json")

model_path = alignnff_fmult()
print("model_path", model_path)
# model_path = "/wrk/knc6/ALINN_FC/ALIGNNFF_DB_FD_mult/temp"
# model_path = "/wrk/knc6/ALINN_FC/ALIGNNFF_DB/temp_new"


def relax(model_path=[]):
    exp_a = []
    aff_a = []
    for i in tqdm(d):
        atoms = Atoms.from_dict(i["atoms"])
        material = i["material"]
        crys = i["Crystal structure"]
        a = i["a"]
        print(material, crys, a)
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            stress_wt=0.3,
            force_multiplier=1,
            force_mult_natoms=False,
        )
        opt, en, fs = ff.optimize_atoms()  # logfile=None)
        print(material, crys, a, opt.lattice.abc[0])
        exp_a.append(float(a))
        aff_a.append(float(opt.lattice.abc[0]))
    info = {}
    info["exp_a"] = exp_a
    info["aff_a"] = aff_a
    dumpjson(data=info, filename="comapre_cubic_lat.json")


relax(model_path=model_path)
