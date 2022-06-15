from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms as JAtoms
from alignn.ff.ff import (
    default_path,
    ev_curve,
    surface_energy,
    vacancy_formation,
    ForceField,
)


def test_alignnff():
    atoms = JAtoms.from_dict(get_jid_data()["atoms"])
    model_path = default_path()
    print("model_path", model_path)
    # atoms = atoms.make_supercell_matrix([2, 2, 2])
    # atoms=atoms.strain_atoms(.05)
    # print(atoms)
    ev = ev_curve(atoms=atoms, model_path=model_path)
    surf = surface_energy(atoms=atoms, model_path=model_path)
    print(surf)
    vac = vacancy_formation(atoms=atoms, model_path=model_path)
    print(vac)

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
    xx = ff.run_npt_nose_hoover(steps=5)
