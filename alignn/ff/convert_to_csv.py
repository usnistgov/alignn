from jarvis.db.jsonutils import loadjson
d=loadjson('defect_db.json')
import pandas as pd
df=pd.DataFrame(d)
df['E_vac']=df['EF']
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
df['bulk_poscar']=df['bulk_atoms'].apply(lambda x: Poscar(Atoms.from_dict(x)).to_string())
df['defective_poscar']=df['defective_atoms'].apply(lambda x: Poscar(Atoms.from_dict(x)).to_string())
df.pop('defective_atoms')
df.pop('bulk_atoms')
print (df.columns)
df.to_csv('vacancy.csv')

