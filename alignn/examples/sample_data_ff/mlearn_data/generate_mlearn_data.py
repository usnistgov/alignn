"""Module for generating mlearn dataset."""
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson
from jarvis.db.jsonutils import loadjson

# Ref: https://github.com/materialsvirtuallab/mlearn
data = loadjson("mlearn/data/Si/training.json")
train_structures = [d["structure"] for d in data]
train_energies = [d["outputs"]["energy"] for d in data]
train_forces = [d["outputs"]["forces"] for d in data]
train_stresses = [d["outputs"]["virial_stress"] for d in data]


data = loadjson("mlearn/data/Si/test.json")
test_structures = [d["structure"] for d in data]
test_energies = [d["outputs"]["energy"] for d in data]
test_forces = [d["outputs"]["forces"] for d in data]
test_stresses = [d["outputs"]["virial_stress"] for d in data]


# For ALIGNN-FF

mem = []
count = 0
for i, j, k, l in zip(
    train_structures, train_energies, train_forces, train_stresses
):
    info = {}
    atoms = pmg_to_atoms(i)
    count += 1
    info["jid"] = str(count)
    info["atoms"] = atoms.to_dict()
    info["total_energy"] = j / atoms.num_atoms
    info["forces"] = k
    info["stresses"] = l
    mem.append(info)
for i, j, k, l in zip(
    test_structures, test_energies, test_forces, train_stresses
):
    info = {}
    count += 1
    info["jid"] = str(count)
    atoms = pmg_to_atoms(i)
    info["atoms"] = atoms.to_dict()
    info["total_energy"] = j / atoms.num_atoms
    info["forces"] = k
    info["stresses"] = l
    mem.append(info)
for i, j, k, l in zip(
    test_structures, test_energies, test_forces, train_stresses
):
    info = {}
    count += 1
    info["jid"] = str(count)
    atoms = pmg_to_atoms(i)
    info["atoms"] = atoms.to_dict()
    info["total_energy"] = j / atoms.num_atoms
    info["forces"] = k
    info["stresses"] = l
    mem.append(info)
print(len(mem), len(train_structures), len(test_structures))

dumpjson(data=mem, filename="id_prop_si.json")
