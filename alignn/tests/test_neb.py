from ase.io import read
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.neb import NEB
from ase.optimize import BFGS
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from alignn.ff.ff import AlignnAtomwiseCalculator, ase_to_atoms
from alignn.ff.ff import (
    default_path,
    ev_curve,
    surface_energy,
    vacancy_formation,
    ForceField,
)

model_path = default_path()
# model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt1_cutoff_8/out"
model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt10_cutoff_8/out"
acalc = AlignnAtomwiseCalculator(path=model_path, filename="best_model.pt")

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100("Al", size=(2, 2, 3))
add_adsorbate(slab, "Au", 1.7, "hollow")
slab.center(axis=2, vacuum=4.0)

# Make sure the structure is correct:
# view(slab)

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
# print(mask)
slab.set_constraint(FixAtoms(mask=mask))
acalc = AlignnAtomwiseCalculator(path=model_path, filename="best_model.pt")
# Use EMT potential:
# slab.calc = EMT()
slab.calc = acalc  # EMT()

# Initial state:
qn = QuasiNewton(slab, trajectory="initial.traj")
qn.run(fmax=0.05)

# Final state:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = QuasiNewton(slab, trajectory="final.traj")
qn.run(fmax=0.05)

initial = read("initial.traj")
final = read("final.traj")

constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial])

images = [initial]
for i in range(3):
    image = initial.copy()
    acalc = AlignnAtomwiseCalculator(path=model_path, filename="best_model.pt")
    # image.calc = EMT()
    image.calc = acalc  # EMT()
    image.set_constraint(constraint)
    images.append(image)

images.append(final)

neb = NEB(images)
neb.interpolate()
qn = BFGS(neb, trajectory="nebff.traj")
qn.run(fmax=0.05)

import matplotlib.pyplot as plt
from ase.neb import NEBTools
from ase.io import read

images = read("nebff.traj@-5:")

# ens=[]
# for i in images:

nebtools = NEBTools(images)

from jarvis.core.atoms import ase_to_atoms

ens = []
for i in nebtools.images:
    jats = ase_to_atoms(i)
    model_path = default_path()
    ff = ForceField(
        jarvis_atoms=jats,
        model_path=model_path,
    )
    en, fs = ff.unrelaxed_atoms()
    # print(en)
    ens.append(en)
print(max(ens) - min(ens))
print(max(ens), min(ens))
# Get the calculated barrier and the energy change of the reaction.
# Ef, dE = nebtools.get_barrier()
