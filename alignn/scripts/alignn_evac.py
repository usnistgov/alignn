"""Modules for making point-defect vacancies."""
# https://arxiv.org/abs/2205.08366
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.thermodynamics.energetics import unary_energy
from alignn.pretrained import get_figshare_model
from jarvis.db.figshare import data
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.jsonutils import dumpjson

# from jarvis.analysis.structure.spacegroup import Spacegroup3D
# from jarvis.db.figshare import get_jid_data
# from jarvis.analysis.defects.vacancy import Vacancy
# from jarvis.analysis.thermodynamics.energetics import unary_energy
# from alignn.pretrained import get_figshare_model
# from jarvis.db.figshare import data
# from jarvis.core.specie import Specie
# from jarvis.core.graphs import Graph
# import glob
# import pprint
# from collections import OrderedDict
# from jarvis.analysis.structure.spacegroup import Spacegroup3D
# from jarvis.core.utils import rand_select
# from jarvis.core.atoms import Atoms
# from alignn.models.alignn import ALIGNN
# from jarvis.db.jsonutils import loadjson
# import numpy as np

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def atom_to_energy(atoms=None, model=None):
    """Get energy for Atoms."""
    g, lg = Graph.atom_dgl_multigraph(atoms)
    out_data = (
        model([g.to(device), lg.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()[0]
    )
    return out_data


model = get_figshare_model("jv_optb88vdw_total_energy_alignn")


def alignn_evac_tot_energy(
    jid="",
    atoms=None,
    model="",
    on_conventional_cell=False,
    enforce_c_size=8,
    extend=1,
    using_wyckoffs=True,
):
    """Get ALIGNN total energy only model based E_vac."""
    if atoms is None:
        dft_3d = data("dft_3d")

        for i in dft_3d:
            if i["jid"] == jid:
                atoms = Atoms.from_dict(i["atoms"])
    # else:
    #    batoms = atoms

    strts = Vacancy(atoms).generate_defects(
        on_conventional_cell=on_conventional_cell,
        enforce_c_size=enforce_c_size,
        extend=extend,
    )
    mem = []
    for j in strts:
        strt = Atoms.from_dict(j.to_dict()["defect_structure"])
        name = (
            str(jid)
            + "_"
            + str(strt.composition.reduced_formula)
            + "_"
            + j.to_dict()["symbol"]
            + "_"
            + j.to_dict()["wyckoff_multiplicity"]
        )
        bulk_atoms = Atoms.from_dict(j.to_dict()["atoms"])

        pred_bulk_energy = atom_to_energy(
            atoms=bulk_atoms, model=model
        )  # * bulk_atoms.num_atoms
        defective_atoms = strt
        # print ('defective_atoms',strt)
        # defective_energy = i["defective_energy"]
        pred_def_energy = (
            atom_to_energy(atoms=defective_atoms, model=model)
            * defective_atoms.num_atoms
        )
        chem_pot = unary_energy(j.to_dict()["symbol"].replace(" ", ""))
        # print('pred_def_energy',pred_def_energy)
        symb = j.to_dict()["symbol"].replace(" ", "")
        # print('chem_pot',symb,unary_energy(symb))
        Ef2 = (
            pred_def_energy
            - (defective_atoms.num_atoms + 1) * pred_bulk_energy
            + chem_pot
        )
        # print (name,Ef2,j.to_dict()["symbol"])
        info = {}
        info["jid"] = jid
        info["symb"] = symb
        info["Ef2"] = Ef2 + 1.3
        info["wyckoff"] = j.to_dict()["wyckoff_multiplicity"]
        mem.append(info)
        print(info)
    name = jid + ".json"
    dumpjson(data=mem, filename=name)
    return mem


if __name__ == "__main__":
    atoms = Atoms.from_poscar("POSCAR")
    mem = alignn_evac_tot_energy(model=model, atoms=atoms)
    print(mem)
# x = []
# for i in glob.glob("*.json"):
#     x.append(i.split(".json")[0])

# for i in dft_3d:
#     if i["jid"] =='JVASP-816':
#         try:
#             mem = jid_ef(
#                 model=model, jid=i["jid"]
#             )  # atoms=Atoms.from_dict(dft_3d[0]['atoms']))
#         except:
#             pass
