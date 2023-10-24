# train_folder_ff.py --root_dir "/mlearn_data/Ni/"
# --config config_ni.json --output_dir=temp_ni --epochs 30
# python ev_phonon_nvt_test.py

from ase import Atom, Atoms as AseAtoms
from jarvis.core.atoms import ase_to_atoms
import numpy as np
import matplotlib.pyplot as plt
from alignn.ff.ff import AlignnAtomwiseCalculator, phonons, ev_curve

model_path = "temp_ni"  # wt10_path() #default_path()
model_path = "temp_ni_radius"
model_path = "temp_ni_radius_cutoff"
model_path = "temp_ni_radius_cutoff_newfunc"
calc = AlignnAtomwiseCalculator(path=model_path)

lattice_params = np.linspace(3.5, 3.8)
lattice_params = np.linspace(3.1, 4.8)
fcc_energies = []
ready = True
for a in lattice_params:
    atoms = AseAtoms(
        [Atom("Ni", (0, 0, 0))],
        cell=0.5
        * a
        * np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
        pbc=True,
    )

    atoms.set_tags(np.ones(len(atoms)))

    atoms.calc = calc

    e = atoms.get_potential_energy()
    fcc_energies.append(e)

plt.plot(lattice_params, fcc_energies)
plt.title("1x1x1")
plt.xlabel("Lattice constant (A)")
plt.ylabel("Total energy (eV)")
plt.savefig("ni.png")

plt.close()

a = 3.51
atoms = AseAtoms(
    [Atom("Cu", (0, 0, 0))],
    cell=0.5
    * a
    * np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
    pbc=True,
)

x, y, eos, kv = ev_curve(atoms=ase_to_atoms(atoms), model_path=model_path)
print(kv)

ase_to_atoms(atoms).write_poscar("POSCAR_Ni")
phonons(atoms=ase_to_atoms(atoms), model_path=model_path, enforce_c_size=3)
cmd = (
    'run_alignn_ff.py --file_path POSCAR_Ni --task="nve_velocity_verlet" '
    + "--timestep=0.1 --md_steps=2000 --temperature_K=305"
    + " --initial_temperature_K=305 --model_path "
    + model_path
)
