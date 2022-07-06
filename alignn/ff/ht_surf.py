import pandas as pd
import re
import numpy as np
from alignn.ff.ff import (
    get_interface_energy,
    vacancy_formation,
    surface_energy,
)
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data, get_jid_data
from jarvis.db.jsonutils import loadjson, dumpjson

d = loadjson("surface_db.json")
mem = []
for i in d:
   jid = i["jid"]
   miller = i["miller"]
   if miller=="1_1_1":
    miller_index = [int(j) for j in i["miller"].split('_')]
    #print(miller,jid)
     
    x = surface_energy(jid=jid, miller_index=miller_index)
    dft_surf = i["surf_en"]
    # elemental so only one value
    alignnff_surf = x[0]["surf_en"]
    symbol = x[0]["name"]
    info = {}
    info["jid"] = jid
    info["dft_surf"] = dft_surf
    info["alignnff_surf"] = alignnff_surf
    info["name"] = symbol
    mem.append(info)
    print(jid, symbol, dft_surf, alignnff_surf)
dumpjson(data=mem,filename="ht_surf.json")
