from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson
import numpy as np
import glob
from jarvis.db.figshare import data
import numpy as np
dft_3d=data("dft_3d")
dft_2d=data("dft_2d")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(6,6))

wad_dft=[]
wad_a=[]
names=[]
for i in glob.glob('JVASP*.json'):
 print(i)
 d=loadjson(i)
 
 dft_wad=d["prev"]
 a_wad=d["Wad"]*-1
 jid_film=d["jid_film"]
 jid_subs=d["jid_subs"]
 miller_film=d["miller_film"]
 miller_subs=d["miller_subs"]
 print(jid_film,jid_subs)
 film_formula=''
 subs_formula=''
 for j in dft_3d:
  if j['jid']==jid_film:
   a=Atoms.from_dict(j['atoms'])
   film_formula=a.composition.reduced_formula
  if j['jid']==jid_subs:
   a=Atoms.from_dict(j['atoms'])
   subs_formula=a.composition.reduced_formula
 if film_formula=='':
  for j in dft_2d:
   if j['jid']==jid_film:
    a=Atoms.from_dict(j['atoms'])
    film_formula=a.composition.reduced_formula

 if subs_formula=='':
  for j in dft_2d:
   if j['jid']==jid_subs:
    a=Atoms.from_dict(j['atoms'])
    subs_formula=a.composition.reduced_formula
 
 print (film_formula,subs_formula,miller_film,miller_subs,dft_wad,a_wad)
 if a_wad>0 and a_wad!=0.02511490598221161:# and a_wad!= 0.1434983058034958:
  wad_dft.append(dft_wad)
  wad_a.append(a_wad)
  formula=film_formula+'/'+subs_formula
  names.append(formula)

wad_dft=np.array(wad_dft,dtype='float')
wad_a=np.array(wad_a,dtype='float')
df=wad_dft-wad_a
order=np.argsort(df)
wad_dft=wad_dft[order]
wad_a=wad_a[order]
print(wad_dft)
print(wad_a)
print('names',np.array(names)[order])
plt.bar(np.arange(len(wad_dft)),wad_dft,width=0.2)
plt.bar(np.arange(len(wad_a))+.2,wad_a,width=0.2)
plt.xticks(np.arange(len(wad_dft)),np.array(names)[order],rotation=45)
plt.ylabel('$W_{adhesion}(Jm^{-2})$')
plt.tight_layout()
plt.savefig('wad.png')
plt.close()
