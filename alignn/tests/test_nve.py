from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from alignn.ff.ff import default_path, AlignnAtomwiseCalculator, ase_to_atoms

# Use Asap for a huge performance increase if it is installed
use_asap = False

if use_asap:
    from asap3 import EMT

    size = 10
else:
    from ase.calculators.emt import EMT

    size = 3


def test_nve(nsteps=5, timestep=0.1):
    # Set up a crystal
    atoms = FaceCenteredCubic(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbol="Cu",
        size=(size, size, size),
        pbc=True,
    )

    model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt10_cutoff/out"
    model_path = (
        "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt10_cutoff_8/out"
    )
    # model_path = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt1_cutoff_8/out"
    model_path = (
        "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa_wt.1_cutoff_8/out"
    )
    # model_path = default_path()
    print("model_path", model_path)
    acalc = AlignnAtomwiseCalculator(path=model_path, filename="best_model.pt")
    # Describe the interatomic interactions with the Effective Medium Theory
    atoms.calc = acalc

    # Set the momenta corresponding to T=300K
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # We want to run MD with constant energy using the VelocityVerlet algorithm.
    dyn = VelocityVerlet(atoms, timestep * units.fs)  # 5 fs time step.

    def printenergy(a):
        """Function to print the potential, kinetic and total energy"""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        print(
            "Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "
            "Etot = %.3feV"
            % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
        )

    # Now run the dynamics
    printenergy(atoms)
    for i in range(nsteps):
        dyn.run(10)
        printenergy(atoms)


test_nve(nsteps=10)
