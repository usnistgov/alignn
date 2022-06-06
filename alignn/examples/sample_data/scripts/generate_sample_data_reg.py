"""Module to generate example dataset."""
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jdata

dft_3d = jdata("dft_3d")
prop = "optb88vdw_bandgap"
max_samples = 50
f = open("id_prop.csv", "w")
count = 0
for i in dft_3d:
    atoms = Atoms.from_dict(i["atoms"])
    jid = i["jid"]
    poscar_name = "POSCAR-" + jid + ".vasp"
    target = i[prop]
    if target != "na":
        atoms.write_poscar(poscar_name)
        f.write(f"{poscar_name},{target:6f}\n")
        count += 1
        if count == max_samples:
            break
f.close()
