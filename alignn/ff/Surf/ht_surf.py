import pandas as pd
import re,os
from joblib import Parallel, delayed
import numpy as np
from alignn.ff.ff import (
    default_path,
    get_interface_energy,
    vacancy_formation,
    surface_energy,
)
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data, get_jid_data
from jarvis.db.jsonutils import loadjson, dumpjson

d = loadjson("surface_db.json")
model_path=default_path()
def make_json(dat=[]):
  i=dat
  fname=i["jid"]+'_'+i["miller"]+'.json'
  if not os.path.exists(fname):
   #try:
    jid = i["jid"]
    miller = i["miller"]
    miller_index = [int(j) for j in i["miller"].split('_')]
    #print(miller,jid)
     
    x = surface_energy(jid=jid, miller_index=miller_index,model_path=model_path)
    dft_surf = i["surf_en"]
    # elemental so only one value
    alignnff_surf = x[0]["surf_en"]
    symbol = x[0]["name"]
    info = {}
    info["jid"] = jid
    info["dft_surf"] = dft_surf
    info["alignnff_surf"] = alignnff_surf
    info["name"] = symbol
    dumpjson(data=info,filename=fname)
    print(jid, symbol, dft_surf, alignnff_surf)
   #except:
   #  pass

Parallel(n_jobs=-1)(delayed(make_json)(i ) for i in d)
