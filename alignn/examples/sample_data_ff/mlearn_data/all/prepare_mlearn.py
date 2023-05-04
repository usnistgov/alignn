"""Module for generating mlearn dataset."""
# for m in Ni Cu Mo Ge Si Li; do wget https://github.com/materialsvirtuallab/mlearn/raw/master/data/${m}/training.json; mv training.json ${m}_train.json; done;
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson
from jarvis.db.jsonutils import loadjson
from pymatgen.core.structure import Structure
import json
from collections import defaultdict
import os
import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress

# Ref: https://github.com/materialsvirtuallab/mlearn

mlearn_dat=[]
els = ["Ni", "Cu", "Mo", "Ge", "Si", "Li"]
train_count=0
val_count=0
test_count=0
for ii in els:
    print(ii)
    name=ii+'_train.json'
    if not os.path.exists(name):
      cmd='wget https://github.com/materialsvirtuallab/mlearn/raw/master/data/'+ii+'/training.json -O '+ii+'_train.json'
      os.system(cmd)

    data = loadjson(name)
    cmd='rm '+ii+'_train.json'
    os.system(cmd)
    train_structures = [d["structure"] for d in data]
    train_energies = [d["outputs"]["energy"] for d in data]
    train_forces = [d["outputs"]["forces"] for d in data]
    train_stresses = np.array(voigt_6_to_full_3x3_stress(np.array([d["outputs"]["virial_stress"] for d in data]))).tolist()
    print ('train_structures',train_energies)
    name=ii+'_test.json'
    if not os.path.exists(name):
     cmd='wget https://github.com/materialsvirtuallab/mlearn/raw/master/data/'+ii+'/test.json -O '+ii+'_test.json'
     os.system(cmd)
    data = loadjson(name)
    cmd='rm '+ii+'_test.json'
    os.system(cmd)
    test_structures = [d["structure"] for d in data]
    test_energies = [d["outputs"]["energy"] for d in data]
    test_forces = [d["outputs"]["forces"] for d in data]
    #test_stresses = [d["outputs"]["virial_stress"] for d in data]
    test_stresses = np.array(voigt_6_to_full_3x3_stress(np.array([d["outputs"]["virial_stress"] for d in data]))).tolist()

    # For ALIGNN-FF
    mem = []
    count = 0
    train_e=defaultdict()
    test_e=defaultdict()
    train_f=defaultdict()
    test_f=defaultdict()
    train_s=defaultdict()
    test_s=defaultdict()


    for i, j, k, l in zip(
        train_structures, train_energies, train_forces, train_stresses
    ):
        k=np.array(k)
        l=np.array(l)
        info = {}
        atoms = pmg_to_atoms(Structure.from_dict(i))
        count += 1
        jid = ii+'-'+str(count)
        info["jid"] = jid
        info["atoms"] = atoms.to_dict()
        info["energy"] = j #/ atoms.num_atoms
        info["total_energy"] = j / atoms.num_atoms
        info["forces"] = (k.tolist())
        info["stresses"] = (l.tolist())
        mem.append(info)
        mlearn_dat.append(info)
        #train[jid]=json.dumps(info)
        train_e[jid]=j
        train_f[jid]=';'.join(map(str,k.flatten()))
        train_s[jid]=';'.join(map(str,l.flatten()))
        train_count+=1
    for i, j, k, l in zip(
        test_structures, test_energies, test_forces, test_stresses
    ):
        k=np.array(k)
        l=np.array(l)
        info = {}
        count += 1
        jid = ii+'-'+str(count)
        info["jid"] = ii+'-'+str(count)
        #atoms = pmg_to_atoms(i)
        atoms = pmg_to_atoms(Structure.from_dict(i))
        info["atoms"] = atoms.to_dict()
        info["energy"] = j #/ atoms.num_atoms
        info["total_energy"] = j / atoms.num_atoms
        info["forces"] = (k.tolist())
        info["stresses"] = (l.tolist())
        #val[jid]=json.dumps(info)
        mem.append(info)
        mlearn_dat.append(info)
        info["jid"]=info["jid"]+"a"
        mlearn_dat.append(info) #For val set
        test_e[jid]=j
        test_f[jid]=';'.join(map(str,k.flatten()))
        test_s[jid]=';'.join(map(str,l.flatten()))
        test_count+=1
    print(len(mem), len(train_structures), len(test_structures))
    dat={}
    dat['train']=train_e
    dat['test']=test_e
    fname='mlearn_'+ii+'_energy.json'
    dumpjson(data=dat, filename=fname)
    cmd='zip '+fname+'.zip '+fname
    os.system(cmd)
    cmd='rm '+fname
    os.system(cmd)

    dat={}
    dat['train']=train_f
    dat['test']=test_f
    fname='mlearn_'+ii+'_forces.json'
    dumpjson(data=dat, filename=fname)
    cmd='zip '+fname+'.zip '+fname
    os.system(cmd)
    cmd='rm '+fname
    os.system(cmd)


    dat={}
    dat['train']=train_s
    dat['test']=test_s
    fname='mlearn_'+ii+'_stresses.json'
    dumpjson(data=dat, filename=fname)
    cmd='zip '+fname+'.zip '+fname
    os.system(cmd)
    cmd='rm '+fname
    os.system(cmd)
#For Figshare
print('train',train_count)
print('test',test_count)
dumpjson(data=mlearn_dat, filename='mlearn.json')
cmd='zip mlearn.json.zip mlearn.json'
os.system(cmd)
cmd='rm mlearn.json'
#os.system(cmd)

