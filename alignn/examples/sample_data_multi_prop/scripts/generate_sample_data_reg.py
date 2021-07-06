from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms

dft_3d = jdata("dft_3d")
props = [
    "optb88vdw_bandgap",
    "formation_energy_peratom",
    "optb88vdw_total_energy",
]
max_samples = 50
f = open("id_multi_prop.csv", "w")
count = 0
for i in dft_3d:
    atoms = Atoms.from_dict(i["atoms"])
    jid = i["jid"]
    poscar_name = "POSCAR-" + jid + ".vasp"
    # cif_name = jid + ".cif"
    target = ""
    for j in props:
        if i[j] != "na":
            target += "," + str(i[j])
    if "na" not in target:
        atoms.write_poscar(poscar_name)
        # atoms.write_cif(cif_name)
        f.write("%s%s\n" % (poscar_name, target))
        count += 1
        if count == max_samples:
            break
f.close()
