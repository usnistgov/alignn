from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms as JAtoms
from alignn.ff.ff import (
    default_path,
    ev_curve,
    surface_energy,
    vacancy_formation,
    ForceField,
    get_interface_energy,
)
from alignn.graphs import Graph, radius_graph_jarvis
from alignn.ff.ff import phonons
from jarvis.core.atoms import ase_to_atoms
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from alignn.ff.ff import (
    AlignnAtomwiseCalculator,
    default_path,
    wt10_path,
    alignnff_fmult,
    fd_path,
    ForceField,
)
from jarvis.io.vasp.inputs import Poscar

# JVASP-25139
pos = """Rb8
1.0
8.534892364405636 0.6983003603741366 -0.0
-3.4905051320748712 7.819743736978101 -0.0
0.0 -0.0 9.899741852856957
Rb
8
Cartesian
-0.48436620907024275 6.0395021169791425 0.0
-0.48395379092975643 6.039257883020857 4.94987
5.528746209070245 2.478537883020856 0.0
5.528333790929757 2.478782116979143 4.94987
1.264246578587533 2.1348318180359995 2.469410532600589
1.2579434214124685 2.1241881819640005 7.419280532600588
3.7864365785875354 6.393851818035999 2.4804594673994105
3.7801334214124656 6.383208181964002 7.430329467399411
"""


# def test_radius_graph_jarvis():
#    atoms = Poscar.from_string(pos).atoms
#    g, lg = radius_graph_jarvis(atoms=atoms)


def test_alignnff():
    atoms = JAtoms.from_dict(get_jid_data()["atoms"])
    atoms = JAtoms.from_dict(
        get_jid_data(dataset="dft_3d", jid="JVASP-1002")["atoms"]
        # get_jid_data(dataset="dft_3d", jid="JVASP-32")["atoms"]
    )
    old_g = Graph.from_atoms(atoms=atoms)
    g, lg = Graph.atom_dgl_multigraph(atoms)
    g, lg = Graph.atom_dgl_multigraph(atoms, neighbor_strategy="radius_graph")
    g, lg = Graph.atom_dgl_multigraph(
        atoms, neighbor_strategy="radius_graph_jarvis"
    )
    model_path = default_path()
    print("model_path", model_path)
    print("atoms", atoms)
    # atoms = atoms.make_supercell_matrix([2, 2, 2])
    # atoms=atoms.strain_atoms(.05)
    ev = ev_curve(atoms=atoms, model_path=model_path)
    # surf = surface_energy(atoms=atoms, model_path=model_path)
    # print('surf',surf)
    # vac = vacancy_formation(atoms=atoms, model_path=model_path)
    # print('vac',vac)

    ff = ForceField(
        jarvis_atoms=atoms,
        model_path=model_path,
    )
    ff.unrelaxed_atoms()
    # sys.exit()
    ff.set_momentum_maxwell_boltzmann()
    xx = ff.optimize_atoms(optimizer="FIRE")
    print("optimized st", xx)
    xx = ff.run_nve_velocity_verlet(steps=5)
    xx = ff.run_nvt_langevin(steps=5)
    xx = ff.run_nvt_andersen(steps=5)
    # xx = ff.run_npt_nose_hoover(steps=5)
    atoms_al = JAtoms.from_dict(
        get_jid_data(dataset="dft_3d", jid="JVASP-1002")["atoms"]
        # get_jid_data(dataset="dft_3d", jid="JVASP-816")["atoms"]
    )
    atoms_al2o3 = JAtoms.from_dict(
        get_jid_data(dataset="dft_3d", jid="JVASP-1002")["atoms"]
        # get_jid_data(dataset="dft_3d", jid="JVASP-32")["atoms"]
    )
    intf = get_interface_energy(
        film_atoms=atoms_al,
        subs_atoms=atoms_al,
        model_path=model_path,
        film_thickness=5,
        subs_thickness=5,
        # film_atoms=atoms_al, subs_atoms=atoms_al2o3, model_path=model_path
    )


def test_phonons():
    atoms = Atoms.from_dict(
        get_jid_data(jid="JVASP-816", dataset="dft_3d")["atoms"]
    )
    ph_path = fd_path()
    ph = phonons(model_path=ph_path, atoms=(atoms))


# test_alignnff()
