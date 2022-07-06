import pandas as pd
import re,os
import numpy as np
from alignn.ff.ff import default_path,get_interface_energy
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data, get_jid_data
from jarvis.db.jsonutils import dumpjson



def str_to_miller(mill="101"):
    x = []
    for i in re.split("(\d)", mill):
        if i != "" and i != '"':
            x.append(int(i))
    return x


def get_atoms(jid=""):
    dft_2d = data("dft_2d")
    for i in dft_2d:
        if i["jid"] == jid:
            atoms = Atoms.from_dict(i["atoms"])
            return atoms
    dft_3d = data("dft_3d")
    for i in dft_3d:
        if i["jid"] == jid:
            atoms = Atoms.from_dict(i["atoms"])
            return atoms

model_path=default_path()
jid_film='JVASP-867'
jid_subs='JVASP-667'
#jid_film='JVASP-816'
#jid_subs='JVASP-32'
miller_film=[1,1,1]
miller_subs=[0,0,1]
film_atoms = get_atoms(jid_film)
subs_atoms = get_atoms(jid_subs)
intf_dat = get_interface_energy(
    film_atoms=film_atoms,
    subs_atoms=subs_atoms,
    film_index=miller_film,
    subs_index=miller_subs,
    seperation=5.0,
    model_path=model_path
)
Wad = intf_dat['interface_energy']
print(
    jid_film,
    jid_subs,
    miller_film,
    miller_subs,
    Wad,
)
print()
print(Atoms.from_dict(intf_dat['optimized_interface']))
info = {}
info['intf_dat']=intf_dat
info["jid_film"] = jid_film
info["jid_subs"] = jid_subs
info["miller_film"] = miller_film
info["miller_subs"] = miller_subs
info["Wad"] = Wad
print(info)
#dumpjson(data=info, filename=fname)
# break
