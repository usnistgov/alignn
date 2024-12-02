from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms as JAtoms
from alignn.ff.ff import (
    default_path,
    ev_curve,
    ForceField,
    get_figshare_model_ff,
)
from alignn.graphs import Graph, radius_graph_jarvis, radius_graph_old
from alignn.ff.ff import phonons, ase_phonon
from jarvis.core.atoms import ase_to_atoms
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from alignn.ff.ff import (
    AlignnAtomwiseCalculator,
    default_path,
    wt10_path,
    ForceField,
)
from jarvis.io.vasp.inputs import Poscar
import os
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

pos = """System
1.0
5.49363 0.0 0.0
-0.0 5.49363 0.0
0.0 0.0 5.49363
Si
8
direct
0.25 0.75 0.25 Si
0.0 0.0 0.5 Si
0.25 0.25 0.75 Si
0.0 0.5 0.0 Si
0.75 0.75 0.75 Si
0.5 0.0 0.0 Si
0.75 0.25 0.25 Si
0.5 0.5 0.5 Si
"""
# def test_radius_graph_jarvis():
#    atoms = Poscar.from_string(pos).atoms
#    g, lg = radius_graph_jarvis(atoms=atoms)


def test_graph_builder():

    atoms = Poscar.from_string(pos).atoms
    old_g = Graph.from_atoms(atoms=atoms)
    g, lg = Graph.atom_dgl_multigraph(atoms)
    g, lg = Graph.atom_dgl_multigraph(atoms, neighbor_strategy="radius_graph")
    g, lg = Graph.atom_dgl_multigraph(
        atoms, neighbor_strategy="radius_graph_jarvis"
    )
    g = radius_graph_old(atoms)


def test_ev():
    atoms = Poscar.from_string(pos).atoms
    model_path = get_figshare_model_ff(
        model_name="v10.30.2024_dft_3d_307k"
    )  # default_path()
    print("model_path", model_path)
    print("atoms", atoms)
    # atoms = atoms.make_supercell_matrix([2, 2, 2])
    # atoms=atoms.strain_atoms(.05)
    ev = ev_curve(atoms=atoms, model_path=model_path, on_relaxed_struct=True)
    # surf = surface_energy(atoms=atoms, model_path=model_path)
    # print('surf',surf)
    # vac = vacancy_formation(atoms=atoms, model_path=model_path)
    # print('vac',vac)


def test_ev():
    atoms = Poscar.from_string(pos).atoms
    # model_path = default_path()
    model_path = get_figshare_model_ff(
        model_name="v10.30.2024_dft_3d_307k"
    )  # default_path()
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
    # xx = ff.run_npt_berendsen(steps=5)
    # xx = ff.run_npt_nose_hoover(steps=5)


def test_phonons():
    atoms = Poscar.from_string(pos).atoms.get_primitive_atoms
    # model_path = default_path()
    model_path = get_figshare_model_ff(
        model_name="v10.30.2024_dft_3d_307k"
    )  # default_path()
    ph = phonons(model_path=model_path, atoms=(atoms))
    ase_phonon(atoms=atoms, model_path=model_path)


def test_qclean():
    cmd = "rm *.pt *.traj *.csv *.json *range"
    os.system(cmd)


# print('test_graph_builder')
# test_graph_builder()
# print('test_ev')
# test_ev()
# print('test_phonons')
# test_phonons()
# test_alignnff()
