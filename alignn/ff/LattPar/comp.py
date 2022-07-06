from jarvis.db.jsonutils import loadjson
from sklearn.metrics import mean_absolute_error as mae
import glob
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.switch_backend('agg')
from jarvis.core.atoms import Atoms
import numpy as np
from jarvis.db.figshare import data
import numpy as np
dft_3d=data("dft_3d")
dft_2d=data("dft_2d")
adft=[]
bdft=[]
cdft=[]
vdft=[]

aal=[]
bal=[]
cal=[]
val=[]
for i in glob.glob("JVASP-*"):
  d=loadjson(i)
  opt_atoms=d['opt_atoms']
  dft_atoms=d['dft_atoms']
  a_alignn = opt_atoms['lattice_mat'][0][0]
  a_dft = dft_atoms['lattice_mat'][0][0]
   
  b_alignn = opt_atoms['lattice_mat'][1][1]
  b_dft = dft_atoms['lattice_mat'][1][1]
  c_alignn = opt_atoms['lattice_mat'][2][2]
  c_dft = dft_atoms['lattice_mat'][2][2]
  jid=d['jid']
  diff_a=abs(a_dft-a_alignn) #/a_dft
  diff_b=abs(b_dft-b_alignn) #/b_dft
  diff_c=abs(c_dft-c_alignn) #/c_dft
  if diff_a>1:
      print (jid,'a',diff_a)
  if diff_b>1:
      print (jid,'b',diff_b)
  if diff_c>1:
      print (jid,'c',diff_c)
  if diff_a<1 and diff_b<1 and diff_c<1:
   adft.append(a_dft)
   bdft.append(b_dft)
   cdft.append(c_dft)

   aal.append(a_alignn)
   bal.append(b_alignn)
   cal.append(c_alignn)
   vdft.append(Atoms.from_dict(dft_atoms).volume)
   val.append(Atoms.from_dict(opt_atoms).volume)
  #print (d['jid'],opt_atoms['lattice_mat'][0][0],dft_atoms['lattice_mat'][0][0])


plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10,6))
the_grid = GridSpec(2,3)
plt.subplot(the_grid[0, 0])
plt.title('(a) a')
plt.scatter(adft,aal,c='red',s=10)
#plt.plot(adft,adft)
plt.plot(np.arange(0,16,1),np.arange(0,16,1),c='black')
plt.xlim([1,15])
plt.ylim([1,15])
plt.xlabel('DFT-a($\AA$)')
plt.ylabel('FF-a($\AA$)')
print ('MAE a',mae(np.array(adft),np.array(aal)))


plt.subplot(the_grid[0, 1])
plt.title('(b) b')
plt.xlim([1,15])
plt.ylim([1,15])
plt.scatter(bdft,bal,c='green',s=10)
#plt.plot(bdft,bdft)
plt.plot(np.arange(0,16,1),np.arange(0,16,1),c='black')
print ('MAE b',mae(np.array(bdft),np.array(bal)))
plt.xlabel('DFT-b($\AA$)')
plt.ylabel('FF-b($\AA$)')

plt.subplot(the_grid[0, 2])
plt.title('(c) c')
plt.xlabel('DFT-c($\AA$)')
plt.ylabel('FF-c($\AA$)')
plt.xlim([1,15])
plt.ylim([1,15])
plt.scatter(cdft,cal,c='blue',s=10)
plt.plot(cdft,cdft,c='black')
print ('MAE c',mae(np.array(cdft),np.array(cal)))



plt.subplot(the_grid[1, 0])
dft_vac=[]
a_vac=[]
for i in glob.glob("/home/knc6/Software/atomwise/alignn/alignn/ff/Vac/JVASP-*"):
  d=loadjson(i)
  opt_atoms=d['alignnff_evac']
  dft_atoms=d['dft_evac']
  jid=d['jid']
  #print (d['jid'],dft_atoms,opt_atoms)
  if abs(opt_atoms-dft_atoms)<2:
     dft_vac.append(dft_atoms)
     a_vac.append(opt_atoms)
  else:
      print('High vac',jid,opt_atoms,dft_atoms)
print ('MAE vac',mae(np.array(dft_vac),np.array(a_vac)))
plt.scatter(dft_vac,a_vac,c='navy',s=10)
plt.plot(np.arange(0,5,1),np.arange(0,5,1),c='black')
plt.xlim([0,4])
plt.ylim([0,4])
plt.xlabel('$E_{vac}$(DFT)(eV)')
plt.ylabel('$E_{vac}$(FF)(eV)')
plt.title('(d) Vacancy')


plt.subplot(the_grid[1, 1])
dft_vac=[]
a_vac=[]
for i in glob.glob("/home/knc6/Software/atomwise/alignn/alignn/ff/Surf/JVASP-*"):
  d=loadjson(i)
  opt_atoms=d['alignnff_surf']
  dft_atoms=d['dft_surf']
  jid=d['jid']
  #print (d['jid'],dft_atoms,opt_atoms)
  if abs(opt_atoms-dft_atoms)<2:
     dft_vac.append(dft_atoms)
     a_vac.append(opt_atoms)
  else:
      print('High surf',jid,opt_atoms,dft_atoms)
print ('MAE surf',mae(np.array(dft_vac),np.array(a_vac)))
plt.scatter(dft_vac,a_vac,c='darkgreen',s=10)
plt.plot(np.arange(0,5,1),np.arange(0,5,1),c='black')
plt.xlim([0,3])
plt.ylim([0,3])
plt.xlabel('$E_{surf}$(DFT)(eV)')
plt.ylabel('$E_{surf}$(FF)(eV)')
plt.title('(d) Surface')

plt.subplot(the_grid[1, 2])
plt.title('(f) Interface')
wad_dft=[]
wad_a=[]
names=[]
for i in glob.glob('/home/knc6/Software/atomwise/alignn/alignn/ff/Intf/JVASP*.json'):
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
plt.xticks(np.arange(len(wad_dft)),np.array(names)[order],rotation=45,fontsize=10)
plt.ylabel('$W_{adhesion}(Jm^{-2})$')

plt.tight_layout(pad=0.5)
plt.savefig('latt_comp.png')
plt.close()
