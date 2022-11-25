from jarvis.core.atoms import Atoms, get_supercell_dims, ase_to_atoms
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
    get_wyckoff_position_operators,
)
from jarvis.core.kpoints import Kpoints3D as Kpoints
import numpy as np
import matplotlib.pyplot as plt
from jarvis.db.figshare import data, get_jid_data
from jarvis.core.atoms import Atoms
from alignn.ff.ff import (
    default_path,
    AlignnAtomwiseCalculator,
    ase_to_atoms,
    ForceField,
)

#%matplotlib inline
def atom_to_energy_forces(atoms=None, model_path=""):
    # """Get energy for Atoms."""
    # strt=atoms.pymatgen_converter)
    # relaxer = Relaxer( relax_cell=False)  # this loads the default model
    # strt=atoms.pymatgen_converter()
    # relax_results = relaxer.relax(strt)
    # final_structure = relax_results['final_structure']
    # final_energy = relax_results['trajectory'].energies[-1] / 2
    # return final_energy,relax_results['trajectory'].forces[-1]

    ff = ForceField(
        jarvis_atoms=atoms,
        model_path=model_path,
    )
    en, fs = ff.unrelaxed_atoms()
    fs = np.array(fs) * -1
    return en, fs


def phonons(
    atoms=None, enforce_c_size=1, line_density=10, model_path="", disp=0.001
):
    """Make Phonon calculation setup."""
    from phonopy import Phonopy
    from phonopy.file_IO import (
        #    parse_FORCE_CONSTANTS,
        write_FORCE_CONSTANTS,
    )

    kpoints = Kpoints().kpath(atoms, line_density=line_density)
    spg = Spacegroup3D(atoms=atoms)  # .spacegroup_data()
    cvn = spg.conventional_standard_structure

    dim = get_supercell_dims(cvn, enforce_c_size=enforce_c_size)
    atoms = cvn.make_supercell([dim[0], dim[1], dim[2]])
    bulk = atoms.phonopy_converter()

    #     Poscar(atoms).write_file("POSCAR")

    # atoms = atoms.make_supercell_matrix([dim[0], dim[1], dim[2]])
    #     Poscar(atoms).write_file("POSCAR-Super.vasp")
    print(bulk)
    phonon = Phonopy(bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]])
    #     print("[Phonopy] Atomic displacements1:", bulk)
    #     print(
    #         "[Phonopy] Atomic displacements2:", phonon, dim[0], dim[1], dim[2]
    #     )
    phonon.generate_displacements(distance=disp)
    disps = phonon.get_displacements()
    #     print("[Phonopy] Atomic displacements3:", disps)
    #     for d in disps:
    #         print("[Phonopy]", d[0], d[1:])
    supercells = phonon.get_supercells_with_displacements()

    # Force calculations by calculator
    set_of_forces = []
    disp = 0
    from ase import Atoms as AseAtoms

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

        energy, forces = atom_to_energy_forces(
            atoms=j_atoms, model_path=model_path
        )
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

    #     print()
    #     print("[Phonopy] Phonon frequencies at Gamma:")
    # print ("kpoints.kpts",kpoints.kpts)
    freqs = []
    for k in kpoints.kpts:
        tmp = []
        for i, freq in enumerate(phonon.get_frequencies(k)):
            # print ("[Phonopy] %3d: %10.5f cm-1" %  (i + 1, freq*33.356)) # THz
            tmp.append(freq * 33.356)
        freqs.append(tmp)
    return freqs


# atoms=Atoms.from_dict(get_jid_data(jid='JVASP-32',dataset='dft_3d')['atoms'])
atoms = Atoms.from_dict(
    get_jid_data(jid="JVASP-1002", dataset="dft_3d")["atoms"]
)
model_path = default_path()
model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt.1_cutoff_8/out"
model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt1_cutoff_8/out"
model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt10_cutoff/out"

freqs = phonons(atoms=atoms, model_path=model_path, disp=0.001)
freqs = np.array(freqs)
for i in range(freqs.shape[1]):
    plt.plot(freqs[:, i], c="b")
plt.savefig("bands.png")
plt.close()
