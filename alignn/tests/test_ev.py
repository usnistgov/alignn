from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import glob
from jarvis.db.jsonutils import loadjson
from jarvis.db.jsonutils import dumpjson
from jarvis.core.atoms import Atoms

plt.switch_backend("agg")
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import get_jid_data
from jarvis.analysis.structure.spacegroup import Spacegroup3D
import numpy as np
from ase.calculators.eam import EAM
from ase.eos import EquationOfState
from alignn.ff.ff import (
    default_path,
    ev_curve,
    surface_energy,
    vacancy_formation,
    ForceField,
    # phonons,
    # phonons3
)

model_path = default_path()
print("model_path", model_path)


def get_vasp_ev(path="/rk2/knc6/EV-curve-rarit/EV-curve-rarit/JVASP-14971_EV"):
    full_path = path + "/*/JVASP*.json"
    energies = []
    volumes = []
    for i in glob.glob(full_path):
        # print (i)
        d = loadjson(i)
        atoms = Atoms.from_dict(d[0]["poscar"])
        natoms = atoms.num_atoms
        fin_energy = d[0]["final_energy"]
        volume = atoms.volume
        # nbs=atoms.get_all_neighbors(r=8)
        # print (min(nbs[0][:,2]),volume,fin_energy)
        energies.append(fin_energy / natoms)
        volumes.append(volume / natoms)
    return energies, volumes


def prepare_dataset():
    mem = {}
    mishin = EAM(potential="Mishin-Ni-Al-Co-2013.eam.alloy")

    dx = np.arange(-0.1, 0.1, 0.01)

    # ni3al
    ni3al = Spacegroup3D(
        Atoms.from_dict(
            get_jid_data(jid="JVASP-14971", dataset="dft_3d")["atoms"]
        )
    ).conventional_standard_structure
    x_ni3al, y_ni3al, eos, kv = ev_curve(ni3al, model_path=model_path, dx=dx)
    vol_ni3al = np.array(eos.v) / ni3al.num_atoms
    y_ni3al = -1 * np.array(y_ni3al) / min(y_ni3al)

    ens_ni3al = []

    for ii, i in enumerate(dx):
        s1 = ni3al.strain_atoms(i)
        s1_ase = s1.ase_converter()
        s1_ase.calc = mishin
        pe = s1_ase.get_potential_energy()
        ens_ni3al.append(pe / s1.num_atoms)
        # vol_ni3al.append(s1.volume/s1.num_atoms)

    ens_ni3al = -1 * np.array(ens_ni3al) / min(ens_ni3al)

    ens_ni3al_vasp, vol_ni3al_vasp = get_vasp_ev(
        path="/rk2/knc6/EV-curve-rarit/EV-curve-rarit/JVASP-14971_EV"
    )
    ens_ni3al_vasp = -1 * np.array(ens_ni3al_vasp) / min(ens_ni3al_vasp)
    info = {}
    info["jid"] = "JVASP-14971"
    info["atoms"] = ni3al.to_dict()
    info["eam"] = [list(ens_ni3al.tolist()), list(vol_ni3al.tolist())]
    info["dft"] = [list(ens_ni3al_vasp.tolist()), list(vol_ni3al_vasp)]
    info["ff_old"] = [list(y_ni3al.tolist()), list(vol_ni3al.tolist())]
    # print(info)
    # dumpjson(data=info,filename='xx.json')

    mem["ni3al"] = info
    # alconi
    alconi = Spacegroup3D(
        Atoms.from_dict(
            get_jid_data(jid="JVASP-108163", dataset="dft_3d")["atoms"]
        )
    ).conventional_standard_structure
    x_alconi, y_alconi, eos, kv = ev_curve(
        alconi, model_path=model_path, dx=dx
    )
    vol_alconi = np.array(eos.v) / s1.num_atoms
    y_alconi = -1 * np.array(y_alconi) / min(y_alconi)
    ens_alconi = []
    for ii, i in enumerate(dx):
        s1 = alconi.strain_atoms(i)
        s1_ase = s1.ase_converter()
        s1_ase.calc = mishin
        pe = s1_ase.get_potential_energy()
        ens_alconi.append(pe / s1.num_atoms)

    ens_alconi = -1 * np.array(ens_alconi) / min(ens_alconi)
    ens_alconi_vasp, vol_alconi_vasp = get_vasp_ev(
        path="/rk2/knc6/EV-curve-rarit/EV-curve-rarit/JVASP-108163_EV"
    )
    ens_alconi_vasp = -1 * np.array(ens_alconi_vasp) / min(ens_alconi_vasp)
    info = {}
    info["jid"] = "JVASP-108163"
    info["atoms"] = alconi.to_dict()
    info["eam"] = [list(ens_alconi.tolist()), list(vol_alconi.tolist())]
    info["dft"] = [
        list(ens_alconi_vasp.tolist()),
        list(np.array(vol_alconi_vasp).tolist()),
    ]
    info["ff_old"] = [list(y_alconi.tolist()), list(vol_alconi.tolist())]
    # mem.append(info)
    mem["alconi"] = info
    # print(info)
    # dumpjson(data=info,filename='xx.json')

    # Feconicr
    farkas = EAM(potential="FeNiCrCoCu-with-ZBL.eam.alloy")
    feconicr = Spacegroup3D(
        Atoms.from_poscar("POSCAR-FeCoNiCr.vasp")
    ).conventional_standard_structure
    x_feconicr, y_feconicr, eos, kv = ev_curve(
        feconicr, model_path=model_path, dx=dx
    )
    vol_feconicr = np.array(eos.v) / feconicr.num_atoms
    y_feconicr = -1 * np.array(y_feconicr) / min(y_feconicr)
    ens_feconicr = []
    for ii, i in enumerate(dx):
        s1 = feconicr.strain_atoms(i)
        s1_ase = s1.ase_converter()
        s1_ase.calc = farkas
        pe = s1_ase.get_potential_energy()
        ens_feconicr.append(pe / s1.num_atoms)

    ens_feconicr = -1 * np.array(ens_feconicr) / min(ens_feconicr)
    ens_feconicr_vasp, vol_feconicr_vasp = get_vasp_ev(
        path="/rk2/knc6/EV-curve-rarit/EV-curve-rarit/JVASP-CrFeCoNi_EV"
    )
    ens_feconicr_vasp = (
        -1 * np.array(ens_feconicr_vasp) / min(ens_feconicr_vasp),
    )
    info = {}
    info["jid"] = "x"
    info["atoms"] = feconicr.to_dict()
    info["eam"] = [list(ens_feconicr.tolist()), list(vol_feconicr.tolist())]
    info["dft"] = [
        list(np.array(ens_feconicr_vasp).tolist()),
        np.array(vol_feconicr_vasp).tolist(),
    ]
    info["ff_old"] = [list(y_feconicr.tolist()), list(vol_feconicr.tolist())]
    # mem.append(info)
    mem["feconicr"] = info
    # print(info)
    # dumpjson(data=info,filename='xx.json')
    ######################## COVALENT #########################

    # SiGe

    # sige = Spacegroup3D(
    #    Atoms.from_dict(
    #        get_jid_data(jid="JVASP-105410", dataset="dft_3d")["atoms"]
    #    )
    # ).conventional_standard_structure
    # x_sige, y_sige, eos, kv = ev_curve(sige, model_path=model_path, dx=dx)
    # vol_sige = np.array(eos.v) / sige.num_atoms

    # GaAs
    # gaas = Spacegroup3D(
    #    Atoms.from_dict(get_jid_data(jid="JVASP-1174", dataset="dft_3d")["atoms"])
    # ).conventional_standard_structure
    # x_gaas, y_gaas, eos, kv = ev_curve(gaas, model_path=model_path, dx=dx)
    # vol_gaas = np.array(eos.v) / gaas.num_atoms

    ########################Ionic####################
    # nacl 23862

    nacl = Atoms.from_dict(
        get_jid_data(jid="JVASP-23862", dataset="dft_3d")["atoms"]
    )  # .conventional_standard_structure

    x_nacl, y_nacl, eos, kv = ev_curve(nacl, model_path=model_path, dx=dx)
    vol_nacl = np.array(eos.v) / nacl.num_atoms
    y_nacl = -1 * np.array(y_nacl) / min(y_nacl)
    ens_nacl_vasp, vol_nacl_vasp = get_vasp_ev(
        path="/rk2/knc6/EV-curve-rarit/EV-curve-rarit/JVASP-23862_EV"
    )
    ens_nacl_vasp = -1 * np.array(ens_nacl_vasp) / min(ens_nacl_vasp)
    info = {}
    info["jid"] = "JVASP-23862"
    info["atoms"] = nacl.to_dict()
    info["eam"] = []  # [list(ens_nacl),list(vol_nacl)]
    info["dft"] = [
        list(ens_nacl_vasp.tolist()),
        list(np.array(vol_nacl_vasp).tolist()),
    ]
    info["ff_old"] = [list(y_nacl.tolist()), list(vol_nacl.tolist())]
    # mem.append(info)
    mem["nacl"] = info
    # print(info)
    # dumpjson(data=info,filename='xx.json')

    # mgo
    # sio2 = Spacegroup3D(
    mgo = Atoms.from_dict(
        get_jid_data(jid="JVASP-116", dataset="dft_3d")["atoms"]
    )  # .conventional_standard_structure

    x_mgo, y_mgo, eos, kv = ev_curve(mgo, model_path=model_path, dx=dx)
    vol_mgo = np.array(eos.v) / mgo.num_atoms
    y_mgo = -1 * np.array(y_mgo) / min(y_mgo)
    ens_mgo_vasp, vol_mgo_vasp = get_vasp_ev(
        path="/rk2/knc6/EV-curve-rarit/EV-curve-rarit/JVASP-116_EV"
    )
    ens_mgo_vasp = -1 * np.array(ens_mgo_vasp) / min(ens_mgo_vasp)
    info = {}
    info["jid"] = "JVASP-116"
    info["atoms"] = mgo.to_dict()
    info["eam"] = []  # [list(ens_mgo),list(vol_mgo)]
    info["dft"] = [
        list(ens_mgo_vasp.tolist()),
        list(np.array(vol_mgo_vasp).tolist()),
    ]
    info["ff_old"] = [list(y_mgo.tolist()), list(vol_mgo.tolist())]
    # mem.append(info)
    mem["mgo"] = info
    # print(info)
    # dumpjson(data=info,filename='xx.json')

    # batio3

    # sion = Atoms.from_dict(
    batio3 = Atoms.from_dict(
        get_jid_data(jid="JVASP-8029", dataset="dft_3d")["atoms"]
    )  # ).conventional_standard_structure

    x_batio3, y_batio3, eos, kv = ev_curve(
        batio3, model_path=model_path, dx=dx
    )
    vol_batio3 = np.array(eos.v) / batio3.num_atoms
    y_batio3 = -1 * np.array(y_batio3) / min(y_batio3)
    ens_batio3_vasp, vol_batio3_vasp = get_vasp_ev(
        path="/rk2/knc6/EV-curve-rarit/EV-curve-rarit/JVASP-8029_EV"
    )
    ens_batio3_vasp = (-1 * np.array(ens_batio3_vasp) / min(ens_batio3_vasp),)
    info = {}
    info["jid"] = "JVASP-8029"
    info["atoms"] = batio3.to_dict()
    info["eam"] = []  # [list(ens_batio3),list(vol_batio3)]
    info["dft"] = [
        list(np.array(ens_batio3_vasp).tolist()),
        list(np.array(vol_batio3_vasp).tolist()),
    ]
    info["ff_old"] = [list(y_batio3.tolist()), list(vol_batio3.tolist())]
    # mem.append(info)
    mem["batio3"] = info
    # print(info)
    # dumpjson(data=info,filename='xx.json')

    dumpjson(data=mem, filename="ev_data.json")
    return mem


# mem = prepare_dataset()
dx = np.arange(-0.1, 0.1, 0.01)
mem = loadjson("ev_data.json")
model_path = default_path()
model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt10_cutoff_8/out"
model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt1_cutoff_8/out"
# model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt.1_cutoff_8/out"
for key, info in mem.items():
    atoms = Atoms.from_dict(info["atoms"])
    x, y, eos, kv = ev_curve(atoms, model_path=model_path, dx=dx)
    vol = np.array(eos.v) / atoms.num_atoms
    y = -1 * np.array(y) / min(y)
    info["ff"] = [y, vol]
    mem[key] = info
##########################################

the_grid = GridSpec(2, 3)
plt.rcParams.update({"font.size": 20})
plt.figure(figsize=(16, 10))
s = 60


plt.subplot(the_grid[0, 0])
key = "ni3al"
vol_ni3al = mem[key]["ff"][1]
y_ni3al = mem[key]["ff"][0]
ens_ni3al = mem[key]["eam"][0]
vol_ni3al_vasp = mem[key]["dft"][1]
ens_ni3al_vasp = mem[key]["dft"][0]
plt.title("(a) $Ni_3Al$")  # +ni3al.composition.reduced_formula)
plt.scatter(vol_ni3al, y_ni3al, s=s, label="ALIGNN-FF")

plt.scatter(vol_ni3al, (ens_ni3al), s=s, label="EAM")
plt.scatter(
    vol_ni3al_vasp,
    ens_ni3al_vasp,
    s=s,
    label="DFT",
    c="limegreen",
)
plt.legend()
plt.xlabel("Volume/atom ($\AA^3$)")
plt.ylabel("Scaled energy (eV/atom)")
plt.tight_layout()
# plt.ylabel('Scaled energy/atom')


plt.subplot(the_grid[0, 1])
# vol_alconi = mem['ni3al']["ff"][1]
# y_ni3al = mem['ni3al']["ff"][0]
# ens_ni3al=mem['ni3al']['eam'][0]
# vol_ni3al_vasp=mem['ni3al']['dft'][1]
# vol_ni3al_vasp=mem['ni3al']['dft'][0]


plt.title("(b) $Al_2CoNi$")  # +alconi.composition.reduced_formula)
key = "alconi"
vol_alconi = mem[key]["ff"][1]
y_alconi = mem[key]["ff"][0]
ens_alconi = mem[key]["eam"][0]
vol_alconi_vasp = mem[key]["dft"][1]
ens_alconi_vasp = mem[key]["dft"][0]

plt.scatter(vol_alconi, (y_alconi), s=s, label="ALIGNN-FF")
plt.scatter(vol_alconi, (ens_alconi), s=s, label="EAM")
plt.scatter(
    vol_alconi_vasp,
    (ens_alconi_vasp),
    s=s,
    label="DFT",
    c="limegreen",
)
plt.legend()
plt.xlabel("Volume/atom ($\AA^3$)")
plt.tight_layout()


plt.subplot(the_grid[0, 2])
key = "feconicr"
vol_feconicr = mem[key]["ff"][1]
y_feconicr = mem[key]["ff"][0]
ens_feconicr = mem[key]["eam"][0]
vol_feconicr_vasp = mem[key]["dft"][1]
ens_feconicr_vasp = mem[key]["dft"][0]
plt.title("(c) CrFeCoNi")  # + feconicr.composition.reduced_formula)
plt.ylabel("Scaled energy (eV/atom)")
plt.scatter(
    vol_feconicr,
    (y_feconicr),
    s=s,
    label="ALIGNN-FF",
)

plt.scatter(
    vol_feconicr,
    (ens_feconicr),
    s=s,
    label="EAM",
)
plt.scatter(
    vol_feconicr_vasp,
    (ens_feconicr_vasp),
    s=s,
    label="DFT",
    c="limegreen",
)
plt.legend()
plt.xlabel("Volume/atom ($\AA^3$)")
plt.tight_layout()


plt.subplot(the_grid[1, 0])
key = "nacl"
vol_nacl = mem[key]["ff"][1]
y_nacl = mem[key]["ff"][0]
# ens_nacl = mem[key]["eam"][0]
vol_nacl_vasp = mem[key]["dft"][1]
ens_nacl_vasp = mem[key]["dft"][0]
plt.title("(d) NaCl")  # + nacl.composition.reduced_formula)
plt.ylabel("Scaled energy (eV/atom)")
plt.scatter(vol_nacl, (y_nacl), s=s, label="ALIGNN-FF")
plt.scatter(
    vol_nacl_vasp,
    ens_nacl_vasp,
    s=s,
    label="DFT",
    c="limegreen",
)
plt.legend()
plt.ylabel("Scaled energy (eV/atom)")
plt.xlabel("Volume/atom ($\AA^3$)")
plt.tight_layout()


plt.subplot(the_grid[1, 1])
key = "mgo"
vol_mgo = mem[key]["ff"][1]
y_mgo = mem[key]["ff"][0]
# ens_mgo = mem[key]["eam"][0]
vol_mgo_vasp = mem[key]["dft"][1]
ens_mgo_vasp = mem[key]["dft"][0]
plt.title("(e) MgO")  # + mgo.composition.reduced_formula)
plt.scatter(vol_mgo, (y_mgo), s=s, label="ALIGNN-FF")
plt.scatter(
    vol_mgo_vasp,
    (ens_mgo_vasp),
    s=s,
    label="DFT",
    c="limegreen",
)
plt.legend()
plt.xlabel("Volume/atom ($\AA^3$)")
plt.tight_layout()

plt.subplot(the_grid[1, 2])
key = "batio3"
vol_batio3 = mem[key]["ff"][1]
y_batio3 = mem[key]["ff"][0]
# ens_batio3 = mem[key]["eam"][0]
vol_batio3_vasp = mem[key]["dft"][1]
ens_batio3_vasp = mem[key]["dft"][0]
plt.title("(f) $BaTiO_3$")  # +sion.composition.reduced_formula)
plt.scatter(vol_batio3, (y_batio3), s=s, label="ALIGNN-FF")
plt.scatter(
    vol_batio3_vasp,
    (ens_batio3_vasp),
    s=s,
    label="DFT",
    c="limegreen",
)
plt.legend()
plt.xlabel("Volume/atom ($\AA^3$)")

# plt.ylabel('Scaled energy/atom')
plt.tight_layout()
plt.savefig("ev_eamtmp2.pdf")
plt.close()
