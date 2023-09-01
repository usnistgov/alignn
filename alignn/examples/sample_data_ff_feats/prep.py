"""Module to prepapre a json file for extra_features model."""
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.atoms import Atoms
import numpy as np

mem = []
d = loadjson("id_prop_old.json")
for i in d:
    info = i
    atoms = Atoms.from_dict(i["atoms"])
    info["extra_features"] = (
        np.array(
            [atoms.lattice.lat_lengths(), atoms.lattice.lat_angles()]
        ).flatten()
    ).tolist()
    mem.append(info)
dumpjson(data=mem, filename="id_prop.json")
