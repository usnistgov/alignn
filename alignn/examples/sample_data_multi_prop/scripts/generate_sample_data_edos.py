"""Module to generate example dataset."""

from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms

dft_3d = jdata("edos_pdos")
max_samples = 50
f = open("id_multi_prop.csv", "w")
count = 0
for i in dft_3d:
    atoms = Atoms.from_dict(i["atoms"])
    jid = i["jid"]
    poscar_name = "POSCAR-" + jid + ".vasp"
    # cif_name = jid + ".cif"
    target = i["edos_up"]
    if target != "na":
        target = ",".join(map(str, target))
        atoms.write_poscar(poscar_name)
        # atoms.write_cif(cif_name)
        f.write("%s,%s\n" % (poscar_name, target))
        count += 1
        if count == max_samples:
            break
f.close()
