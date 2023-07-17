from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from alignn.ff.ff import (
    # default_path,
    ev_curve,
    AlignnAtomwiseCalculator,
    alignnff_fmult,
)


model_path = alignnff_fmult()

stress_wt = 0.1
calc = AlignnAtomwiseCalculator(
    path=model_path,
    force_mult_natoms=False,
    force_multiplier=1,
    stress_wt=stress_wt,
)


print("model_path", model_path)
# Si
si = ["JVASP-1002", "JVASP-91933", "JVASP-25369", "JVASP-25368"]
nmsi = ["DiamondCubic", "X", "X", "X"]
on_relaxed_struct = True
# dft_3d=data('dft_3d')
memsi = []
for i, j in zip(si, nmsi):
    # try:

    dat = get_jid_data(jid=i, dataset="dft_3d")
    atoms = Spacegroup3D(
        Atoms.from_dict(dat["atoms"])
    ).conventional_standard_structure
    print()
    print(dat["jid"], j, dat["bulk_modulus_kv"], atoms.num_atoms)
    x = ev_curve(
        atoms,
        stress_wt=stress_wt,
        model_path=model_path,
        on_relaxed_struct=on_relaxed_struct,
    )
    info = {}
    info["data"] = x
    info["jid"] = i
    info["atoms"] = atoms
    memsi.append(info)

# except :
#    pass
# MoS2

mote2 = ["JVASP-28733", "JVASP-28413", "JVASP-58505"]
nmmote2 = ["2H", "1T", "cubic"]
memmote2 = []
for i, j in zip(mote2, nmmote2):
    # try:

    dat = get_jid_data(jid=i, dataset="dft_3d")
    atoms = Spacegroup3D(
        Atoms.from_dict(dat["atoms"])
    ).conventional_standard_structure
    print()
    print(dat["jid"], j, dat["bulk_modulus_kv"], atoms.num_atoms)
    x = ev_curve(
        atoms,
        model_path=model_path,
        stress_wt=stress_wt,
        on_relaxed_struct=on_relaxed_struct,
    )
    info = {}
    info["data"] = x
    info["jid"] = i
    info["atoms"] = atoms
    memmote2.append(info)

# except :
#    pass
# Ni3Al

ni3al = ["JVASP-14971", "JVASP-99749", "JVASP-11979"]
nmni3al = ["L12", "Hex.", "FCC"]

# dft_3d=data('dft_3d')
memni3al = []
for i, j in zip(ni3al, nmni3al):
    # try:

    dat = get_jid_data(jid=i, dataset="dft_3d")
    atoms = Spacegroup3D(
        Atoms.from_dict(dat["atoms"])
    ).conventional_standard_structure
    print()
    print(dat["jid"], j, dat["bulk_modulus_kv"], atoms.num_atoms)
    x = ev_curve(
        atoms,
        model_path=model_path,
        stress_wt=stress_wt,
        on_relaxed_struct=on_relaxed_struct,
    )
    info = {}
    info["data"] = x
    info["jid"] = i
    info["atoms"] = atoms
    memni3al.append(info)

# except :
#    pass
# SiO2
sio2 = ["JVASP-58349", "JVASP-34674", "JVASP-34656"]
names_sio2 = ["alpha-QTZ", "alpha-Trid", "alpha-Cryst"]

mem = []
# try:
for i, j in zip(sio2, names_sio2):
    dat = get_jid_data(jid=i, dataset="dft_3d")
    atoms = Atoms.from_dict(dat["atoms"])
    print()
    print(dat["jid"], j, dat["bulk_modulus_kv"], atoms.num_atoms)
    x = ev_curve(
        atoms,
        model_path=model_path,
        stress_wt=stress_wt,
        on_relaxed_struct=on_relaxed_struct,
    )
    info = {}
    info["data"] = x
    info["jid"] = i
    info["atoms"] = atoms
    mem.append(info)

# except :
#    pass


plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 12))
the_grid = GridSpec(3, 2)
plt.clf()
plt.subplot(the_grid[0, 0])
plt.title("(a) Si")
nmsi = ["Diam.Cub.", "Mono.", "Cubic2", "Cubic3"]
for i, j in zip(memsi, nmsi):
    # if j not in ['Cubic3']:
    energies = np.array(i["data"][2].e) / i["atoms"].num_atoms
    vols = i["data"][2].v / i["atoms"].num_atoms
    print(i["jid"], min(i["data"][2].e) / (i["atoms"].num_atoms), j)
    plt.plot(vols, energies, "-*", label=j)
plt.legend(loc="upper right")
plt.ylim([-4.7, -3.4])
plt.xlabel("Volume/atom($A^3$)")
plt.ylabel("Energy/atom(eV)")
# plt.xlim(12,30)


plt.subplot(the_grid[0, 1])
plt.title("(b) SiO$_2$")
nm = [
    "$\\alpha$-Qtz",
    "beta-QTZ",
    "$\\alpha$-Trid",
    "beta-Trid",
    "$\\alpha$-Cryst",
    "$\\beta$-Cryst",
    "coesite",
    "stishovite",
]
nm = ["alpha-QTZ", "alpha-Trid", "alpha-Cryst"]
for i, j in zip(mem, nm):
    # if j not in ['beta-QTZ','coesite',"stishovite",'beta-Trid']:
    energies = np.array(i["data"][2].e) / i["atoms"].num_atoms
    vols = i["data"][2].v / i["atoms"].num_atoms
    print(i["jid"], min(i["data"][2].e) / (i["atoms"].num_atoms), j)
    plt.plot(vols, energies, "-*", label=j)
plt.legend(loc="lower right")
plt.ylim([-6.4, -6])
plt.xlabel("Volume/atom($A^3$)")
plt.ylabel("Energy/atom(eV)")

plt.subplot(the_grid[1, 0])
plt.title("(c) Ni$_3$Al")
for i, j in zip(memni3al, nmni3al):
    energies = np.array(i["data"][2].e) / i["atoms"].num_atoms
    vols = i["data"][2].v / i["atoms"].num_atoms
    print(i["jid"], min(i["data"][2].e) / (i["atoms"].num_atoms), j)
    plt.plot(vols, energies, "-*", label=j)
plt.legend(loc="upper right")
# plt.ylim([-5.3,-5])
plt.xlabel("Volume/atom($A^3$)")
plt.ylabel("Energy/atom(eV)")

plt.subplot(the_grid[1, 1])
plt.title("(d) MoS$_2$")
for i, j in zip(memmote2, nmmote2):
    energies = np.array(i["data"][2].e) / i["atoms"].num_atoms
    vols = i["data"][2].v / i["atoms"].num_atoms
    print(i["jid"], min(i["data"][2].e) / (i["atoms"].num_atoms), j)
    plt.plot(vols, energies, "-*", label=j)
plt.legend(loc="upper right")
plt.ylim([-5.3, -5])
plt.xlabel("Volume/atom($A^3$)")
plt.ylabel("Energy/atom(eV)")


plt.tight_layout()
plt.savefig("ev.png")
plt.close()
