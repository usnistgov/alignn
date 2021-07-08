"""Module to check if V_Ef is approx. correct."""
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn import ALIGNN

# from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.figshare import get_jid_data
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.thermodynamics.energetics import unary_energy

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


def get_defect_form_en(
    jid="JVASP-1002",
    model_path="JV15/jv_optb88vdw_total_energy_alignn/checkpoint_300.pt",
    dataset="dft_3d",
):
    """Predict defect formation energy ???."""
    model = ALIGNN()
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    # model=torch.load('checkpoint_250.pt')['model']
    model.to(device)
    model.eval()

    atoms = Atoms.from_dict(get_jid_data(jid=jid, dataset=dataset)["atoms"])
    bulk_en_pa = atom_to_energy(atoms=atoms, model=model)  # *atoms.num_atoms

    strts = Vacancy(atoms).generate_defects(
        on_conventional_cell=False, enforce_c_size=8, extend=1
    )
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
        print(name)
        def_energy = atom_to_energy(atoms=strt, model=model) * strt.num_atoms
        chem_pot = unary_energy(j.to_dict()["symbol"])
        Ef = def_energy - (strt.num_atoms + 1) * bulk_en_pa + chem_pot
        print(
            j.to_dict()["symbol"],
            Ef,
            bulk_en_pa,
            def_energy,
            atoms.num_atoms,
            chem_pot,
        )


get_defect_form_en()
