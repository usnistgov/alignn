from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from alignn.ff.ff import (
    #    default_path,
    ev_curve,
    AlignnAtomwiseCalculator,
    alignnff_fmult,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


model_path = alignnff_fmult()
on_relaxed_struct = (True,)
stress_wt = 0.1
calc = AlignnAtomwiseCalculator(
    path=model_path,
    force_mult_natoms=False,
    force_multiplier=1,
    stress_wt=stress_wt,
)

jids = [
    "JVASP-14971",
    "JVASP-108163",
    # "JVASP-23862",
    "JVASP-116",
    "JVASP-8029",
    "JVASP-108871",
]
# jids=jids[0:3]
plt.rcParams.update({"font.size": 14})
plt.figure(figsize=(15, 4))
nmats = len(jids)
print("nmats", nmats)
the_grid = GridSpec(1, nmats)

for ii, i in enumerate(jids):
    dat = get_jid_data(jid=i, dataset="dft_3d")
    atoms = Spacegroup3D(
        Atoms.from_dict(dat["atoms"])
    ).conventional_standard_structure
    x = ev_curve(
        atoms,
        stress_wt=stress_wt,
        model_path=model_path,
        on_relaxed_struct=on_relaxed_struct,
    )
    energies = np.array(x[2].e) / atoms.num_atoms
    vols = x[2].v / atoms.num_atoms
    formula = atoms.composition.reduced_formula
    plt.subplot(the_grid[ii])
    plt.title(formula)
    if ii == 0:
        plt.ylabel("E(eV/atom)")

    plt.plot(vols, energies, "-*", label=i)
    plt.xlabel("V")
    print("Formula,DFT,FF", formula, dat["bulk_modulus_kv"], x[-1])
plt.tight_layout()
plt.savefig("ev_chem.png")
plt.close()
