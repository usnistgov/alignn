from ase.md import MDLogger
from jarvis.core.atoms import Atoms as JarvisAtoms
import os
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.io import Trajectory
from ase import Atoms as AseAtoms
import matplotlib.pyplot as plt
from jarvis.analysis.thermodynamics.energetics import unary_energy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.gpmin.gpmin import GPMin
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.constraints import ExpCellFilter
from ase.eos import EquationOfState
from ase.units import kJ
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.md import VelocityVerlet
from ase import units
from ase.md import Langevin
from ase.md.npt import NPT
from ase.md.andersen import Andersen
import ase.calculators.calculator
from ase.stress import full_3x3_to_voigt_6_stress
import torch
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
from alignn.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from jarvis.analysis.defects.vacancy import Vacancy
import numpy as np
from alignn.pretrained import get_prediction
from jarvis.db.figshare import get_jid_data

# from jarvis.analysis.defects.surface import Surface
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
from jarvis.analysis.interface.zur import make_interface
from jarvis.analysis.defects.surface import Surface

# from jarvis.core.kpoints import Kpoints3D as Kpoints
from jarvis.core.atoms import get_supercell_dims
from alignn.ff.ff import (
    default_path,
    AlignnAtomwiseCalculator,
    ase_to_atoms,
    ForceField,
)


def phonons(
    atoms=None,
    enforce_c_size=8,
    line_density=5,
    model_path=".",
    model_filename="best_model.pt",
    on_relaxed_struct=False,
):
    """Make Phonon calculation setup."""

    if on_relaxed_struct:
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            model_filename=model_filename,
        )
        atoms, en, fs = ff.optimize_atoms()
    from phonopy import Phonopy
    from phonopy.file_IO import (
        #    parse_FORCE_CONSTANTS,
        write_FORCE_CONSTANTS,
    )

    # kpoints = Kpoints().kpath(atoms, line_density=line_density)
    spg = Spacegroup3D(atoms=atoms)  # .spacegroup_data()
    cvn = spg.conventional_standard_structure
    dim = get_supercell_dims(cvn, enforce_c_size=enforce_c_size)
    atoms = cvn.make_supercell([dim[0], dim[1], dim[2]])
    bulk = atoms.phonopy_converter()
    phonon = Phonopy(bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]])
    phonon.generate_displacements(distance=0.03)
    print("Len dis", len(phonon.supercells_with_displacements))
    # disps = phonon.get_displacements()
    supercells = phonon.get_supercells_with_displacements()
    # Force calculations by calculator
    set_of_forces = []
    disp = 0

    for scell in supercells:
        ase_atoms = AseAtoms(
            symbols=scell.get_chemical_symbols(),
            scaled_positions=scell.get_scaled_positions(),
            cell=scell.get_cell(),
            pbc=True,
        )
        j_atoms = ase_to_atoms(ase_atoms)
        disp = disp + 1

        # parameters["control_file"] = "run0.mod"

        ff = ForceField(
            jarvis_atoms=j_atoms,
            model_path=model_path,
            model_filename=model_filename,
        )
        st, energy, forces = ff.optimize_atoms(optimize_lattice=False)
        # forces = forces * -1
        # print("forces=", forces)
        drift_force = forces.sum(axis=0)
        # print("drift forces=", drift_force)
        # Simple translational invariance
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)
    phonon.produce_force_constants(forces=set_of_forces)
    write_FORCE_CONSTANTS(
        phonon.get_force_constants(), filename="FORCE_CONSTANTS"
    )
    phonon.run_mesh([20, 20, 20])
    phonon.run_total_dos()
    tdos = phonon._total_dos

    # print('tods',tdos._frequencies.shape)
    freqs, ds = tdos.get_dos()
    # print('tods',tdos.get_dos())
    dosfig = phonon.plot_total_dos()
    dosfig.savefig("dos1.png")
    dosfig.close()

    plt.plot(freqs, ds)
    plt.close("dos2.png")


atoms = JarvisAtoms.from_dict(
    get_jid_data(jid="JVASP-1002", dataset="dft_3d")["atoms"]
)
model_path = default_path()
model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt10_cutoff/out"
phonons(atoms=atoms, model_path=model_path)
