import pandas as pd
from jarvis.db.figshare import data
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices)
from jarvis.core.atoms import Atoms
from alignn.ff.ff import (
    default_path,
    ev_curve,
    surface_energy,
    vacancy_formation,
    ForceField,
)
from jarvis.db.jsonutils import loadjson,dumpjson
import os
import re
import numpy as np
from alignn.ff.ff import default_path,get_interface_energy, vacancy_formation
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data, get_jid_data
from jarvis.db.jsonutils import loadjson, dumpjson
jids_elemnts= ['JVASP-88846', 'JVASP-987', 'JVASP-981', 'JVASP-25388', 'JVASP-984', 'JVASP-834', 'JVASP-14604', 'JVASP-837', 'JVASP-840', 'JVASP-25379', 'JVASP-25144', 'JVASP-14744', 'JVASP-890', 'JVASP-888', 'JVASP-14622', 'JVASP-969', 'JVASP-972', 'JVASP-25254', 'JVASP-25407', 'JVASP-961', 'JVASP-958', 'JVASP-963', 'JVASP-14832', 'JVASP-966', 'JVASP-25125', 'JVASP-802', 'JVASP-25273', 'JVASP-25167', 'JVASP-919', 'JVASP-25114', 'JVASP-922', 'JVASP-949', 'JVASP-95268', 'JVASP-79561', 'JVASP-1056', 'JVASP-14612', 'JVASP-102277', 'JVASP-943', 'JVASP-931', 'JVASP-934', 'JVASP-937', 'JVASP-21193', 'JVASP-946', 'JVASP-25142', 'JVASP-828', 'JVASP-33718', 'JVASP-1011', 'JVASP-25250', 'JVASP-25213', 'JVASP-1002', 'JVASP-14601', 'JVASP-14812', 'JVASP-14837', 'JVASP-996', 'JVASP-993', 'JVASP-7804', 'JVASP-858', 'JVASP-25104', 'JVASP-25180', 'JVASP-852', 'JVASP-25248', 'JVASP-1035', 'JVASP-864', 'JVASP-861', 'JVASP-867', 'JVASP-910', 'JVASP-25117', 'JVASP-25337', 'JVASP-916', 'JVASP-1026', 'JVASP-1029', 'JVASP-25210', 'JVASP-1017', 'JVASP-1020', 'JVASP-1014', 'JVASP-21197', 'JVASP-870', 'JVASP-895', 'JVASP-14725', 'JVASP-1050', 'JVASP-810', 'JVASP-14606', 'JVASP-901', 'JVASP-816', 'JVASP-14603', 'JVASP-819', 'JVASP-825', 'JVASP-898', 'JVASP-21195']

model_path=default_path()
vac_db=loadjson('vacancy_db.json')
jids=list(set(jids_elemnts+[i['jid'] for i in vac_db]))
print ('jids=',len(jids))
mem = []
from joblib import Parallel, delayed

#import ray
#ray.init()
#jids_put=ray.put(jids)
d=data('dft_3d')

# @ray.remote
def make_json(jid=''):
 for i in d:
  name=i['jid']+'.json'
  if i['jid'] == jid and not os.path.exists(name) and i['jid'] not in ['JVASP-107386','JVASP-60771']:
   try:
     print ('HERE',i['jid'])
     atoms=Atoms.from_dict(i['atoms'])
     atoms = Spacegroup3D(atoms).conventional_standard_structure
     ff = ForceField(jarvis_atoms=atoms,model_path=model_path)
     opt_atoms,en,fs = ff.optimize_atoms(optimizer="FIRE",optimize_lattice=True)
     print(i['jid'],atoms.lattice_mat[0][0],opt_atoms.lattice_mat[0][0])
     opt_atoms=opt_atoms.to_dict()
     info={}
     info['jid']=i['jid']
     info['opt_atoms']=opt_atoms
     info['dft_atoms']=atoms.to_dict()
     info['energy']=en
     dumpjson(filename=name,data=info)
     print()
   except:
     pass
 return jid


Parallel(n_jobs=-1)(delayed(make_json)(i ) for i in jids)
"""
make_json.remote(jids_put)
result_ids = []
for i in jids:
 result_ids.append(make_json.remote(i))
results = ray.get(result_ids)  
"""
