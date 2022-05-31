"""Module to run ALIGNN-FF."""
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import ExpCellFilter
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin

# from ase.optimize.optimize import Optimizer
# from ase.io import Trajectory
# from ase.neighborlist import NeighborList
# from ase import Atoms
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from alignn.graphs import Graph

# from alignn.config import TrainingConfig
from jarvis.core.atoms import Atoms as JAtoms

# from jarvis.db.jsonutils import loadjson
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
import torch

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def ase_to_atoms(ase_atoms=""):
    """Convert ase structure to jarvis.core.Atoms."""
    return JAtoms(
        lattice_mat=ase_atoms.get_cell(),
        elements=ase_atoms.get_chemical_symbols(),
        coords=ase_atoms.get_positions(),
        cartesian=True,
    )


def load_model(
    model_path="",
    graphwise_weight=1,
    gradwise_weight=10,
    stresswise_weight=0.1,
    atomwise_weight=0,
    alignn_layers=4,
    gcn_layers=4,
    atom_input_features=92,
    edge_input_features=80,
    triplet_input_features=40,
    embedding_features=64,
    hidden_features=256,
    output_features=1,
    grad_multiplier=-1,
    calculate_gradient=True,
    atomwise_output_features=3,
):
    """Load ALIGNN_FF model."""
    model = ALIGNNAtomWise(
        ALIGNNAtomWiseConfig(
            name="alignn_atomwise",
            graphwise_weight=graphwise_weight,
            gradwise_weight=gradwise_weight,
            stresswise_weight=stresswise_weight,
            atomwise_weight=atomwise_weight,
            alignn_layers=alignn_layers,
            gcn_layers=gcn_layers,
            atom_input_features=atom_input_features,
            edge_input_features=edge_input_features,
            triplet_input_features=triplet_input_features,
            embedding_features=embedding_features,
            hidden_features=hidden_features,
            output_features=output_features,
            grad_multiplier=grad_multiplier,
            calculate_gradient=calculate_gradient,
            atomwise_output_features=atomwise_output_features,
        )
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def atom_to_energy_forces(
    atoms=None,
    model=None,
    cutoff=8.0,
    max_neighbors=12,
    multiply_energy_natoms=True,
):
    """Get energy,force etc. for Atoms."""
    g, lg = Graph.atom_dgl_multigraph(
        atoms, cutoff=cutoff, max_neighbors=max_neighbors
    )
    result = model([g, lg])
    energy = result["out"].detach().numpy()
    if multiply_energy_natoms:
        energy = energy * atoms.num_atoms
    forces = result["grad"].detach().numpy()
    stress = result["stress"].detach().numpy() * -1
    return energy, forces, stress


# def myprint():
#    print(
#        f"time={dyn.get_time() / units.fs: 5.0f} fs "
#        + f"T={ase_atoms.get_temperature(): 3.0f} K"
#    )


class ALIGNNFF_Calculator(Calculator):
    """Module for ALIGNN-FF ASE Calculator."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model=None,
        model_path="",
        cutoff=8.0,
        max_neighbors=12,
        multiply_energy_natoms=True,
        graphwise_weight=1,
        gradwise_weight=10,
        stresswise_weight=0.1,
        atomwise_weight=0,
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92,
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=256,
        output_features=1,
        grad_multiplier=-1,
        atomwise_output_features=3,
        calculate_gradient=True,
        **kwargs,
    ):
        """Initialize class."""
        Calculator.__init__(self, **kwargs)
        self.model = model
        if model is None:
            self.model = load_model(
                model_path=model_path,
                graphwise_weight=graphwise_weight,
                gradwise_weight=gradwise_weight,
                stresswise_weight=stresswise_weight,
                atomwise_weight=atomwise_weight,
                alignn_layers=alignn_layers,
                gcn_layers=gcn_layers,
                atom_input_features=atom_input_features,
                edge_input_features=edge_input_features,
                triplet_input_features=triplet_input_features,
                embedding_features=embedding_features,
                hidden_features=hidden_features,
                output_features=output_features,
                grad_multiplier=grad_multiplier,
                calculate_gradient=calculate_gradient,
                atomwise_output_features=atomwise_output_features,
            )
        print("Model=", model)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.multiply_energy_natoms = multiply_energy_natoms
        # self.compute_stress = compute_stress

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        """Run ASE calculation."""
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        # natoms = len(self.atoms)

        jatoms = ase_to_atoms(self.atoms)
        energy, forces, stress = atom_to_energy_forces(
            atoms=jatoms,
            model=self.model,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            multiply_energy_natoms=self.multiply_energy_natoms,
        )

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress


class TrajLogger(object):
    """Module for logging trajectory."""

    def __init__(
        self,
        ase_atoms=None,
        energies=[],
        forces=[],
        stresses=[],
        cart_coords=[],
        lattice_mats=[],
        elements=[],
    ):
        """Initialize class."""
        self.ase_atoms = ase_atoms
        self.energies = energies
        self.forces = forces
        self.stresses = stresses
        self.cart_coords = cart_coords
        self.lattice_mats = lattice_mats
        self.elements = elements

    def __call__(self):
        """Call module."""
        self.energies.append(self.ase_atoms.get_potential_energy())
        self.forces.append(self.ase_atoms.get_forces())
        self.stresses.append(self.ase_atoms.get_stress())
        self.cart_coords.append(self.ase_atoms.get_positions())
        self.lattice_mats.append(self.ase_atoms.get_cell()[:])
        self.elements.append(self.ase_atoms.get_chemical_symbols())

    def to_dict(self):
        """Return a dictionary."""
        info = {}
        info["energies"] = self.energies
        info["forces"] = self.forces
        info["stresses"] = self.stresses
        info["cart_coords"] = self.cart_coords
        info["lattice_mats"] = self.lattice_mats
        info["elements"] = self.elements
        return info


class OptimizeAtoms(object):
    """Module to optimize/relax atomic structure."""

    def __init__(
        self,
        jarvis_atoms=None,
        optimizer="BFGS",
        optimize_lattice=True,
        calculator=None,
        force_tolerance=0.1,
        nsteps=500,
        interval=1,
    ):
        """Intialize class."""
        self.jarvis_atoms = jarvis_atoms
        self.optimizer = optimizer
        self.optimize_lattice = optimize_lattice
        self.calculator = calculator
        self.force_tolerance = force_tolerance
        self.nsteps = nsteps
        self.interval = interval

    def optimize(self):
        """Optimize atoms."""
        available_ase_optimizers = {
            "BFGS": BFGS,
            "LBFGS": LBFGS,
            "LBFGSLineSearch": LBFGSLineSearch,
            "FIRE": FIRE,
            "MDMin": MDMin,
            "SciPyFminCG": SciPyFminCG,
            "SciPyFminBFGS": SciPyFminBFGS,
            "BFGSLineSearch": BFGSLineSearch,
        }
        optimizer = available_ase_optimizers.get(self.optimizer, None)
        if optimizer is None:
            raise ValueError("Check optimizer", optimizer)

        ase_atoms = self.jarvis_atoms.ase_converter()
        ase_atoms.set_calculator(self.calculator)
        traj_logger = TrajLogger(ase_atoms=ase_atoms)

        optimizer = optimizer(ase_atoms)
        if self.optimize_lattice:
            ase_atoms = ExpCellFilter(ase_atoms)
        optimizer.attach(traj_logger, interval=self.interval)
        optimizer.run(fmax=self.force_tolerance, steps=self.nsteps)
        traj_logger()
        if isinstance(ase_atoms, ExpCellFilter):
            ase_atoms = ase_atoms.atoms
        return ase_to_atoms(ase_atoms)


class RunMD(object):
    """Module to run MD with an ensemble."""

    def __init__(
        self,
        jarvis_atoms=None,
        calculator=None,
        ensemble="nvt",
        temperature=300,
        pressure=1.01325,
        timestep=0.1,
        compressibility=0.5,
        traj_file="md.traj",
        logfile="md.log",
        interval=1,
        nsteps=1000,
        append_trajectory=False,
        max_boltz_initial_temp=10,
    ):
        """Initialize class."""
        self.jarvis_atoms = jarvis_atoms
        self.calculator = calculator
        self.ensemble = ensemble.lower()
        self.temperature = temperature
        self.pressure = pressure * units.bar
        self.timestep = timestep * units.fs
        self.traj_file = traj_file
        self.logfile = logfile
        self.interval = interval
        self.nsteps = nsteps
        self.append_trajectory = append_trajectory
        self.max_boltz_initial_temp = max_boltz_initial_temp
        self.compressibility = compressibility

    def run(self):
        """Run MD."""
        ase_atoms = self.jarvis_atoms.ase_converter()
        ase_atoms.set_calculator(self.calculator)
        if self.max_boltz_initial_temp is not None:
            MaxwellBoltzmannDistribution(
                ase_atoms, self.max_boltz_initial_temp * units.kB
            )
            print("self.max_boltz_initial_temp", self.max_boltz_initial_temp)
        if self.ensemble == "nvt":
            print("self.temperature", self.temperature)
            taut = self.nsteps * self.timestep
            dyn = NVTBerendsen(
                ase_atoms,
                self.timestep,
                self.temperature,
                taut=taut,
                trajectory=self.traj_file,
                logfile=self.logfile,
                loginterval=self.interval,
                append_trajectory=self.append_trajectory,
            )

            def myprint():
                """Print info."""
                print(
                    f"time={dyn.get_time() / units.fs: 5.0f} fs "
                    + f"T={ase_atoms.get_temperature(): 3.0f} K"
                )

            dyn.attach(myprint, interval=self.interval)
            dyn.set_temperature(self.temperature)
            dyn.run(self.nsteps)
        elif self.ensemble == "npt":
            taut = self.nsteps * self.timestep
            taup = self.nsteps * self.timestep
            dyn = NPTBerendsen(
                ase_atoms,
                self.timestep,
                temperature_K=self.temperature,
                pressure_au=self.pressure,
                taut=taut,
                taup=taup,
                compressibility_au=self.compressibility,
                trajectory=self.traj_file,
                logfile=self.logfile,
                loginterval=self.interval,
                append_trajectory=self.append_trajectory,
            )

            def myprint():
                """Print info."""
                print(
                    f"time={dyn.get_time() / units.fs: 5.0f} fs "
                    + f"T={ase_atoms.get_temperature(): 3.0f} K"
                    + f" P={dyn.get_pressure(): 3.0f} kbar"
                )

            dyn.attach(myprint, interval=self.interval)
            dyn.set_temperature(self.temperature)
            dyn.run(self.nsteps)
        else:
            raise NotImplementedError(
                "Ensemble not implemented", self.ensemble
            )


"""
if __name__ == "__main__":
    from jarvis.db.figshare import get_jid_data

    atoms = JAtoms.from_dict(get_jid_data()["atoms"])
    print(atoms)
    model_path = (
        "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa/out/best_model.pt"
    )
    calculator = ALIGNNFF_Calculator(model_path=model_path)
    opt = OptimizeAtoms(
        optimizer="BFGS", calculator=calculator, jarvis_atoms=atoms
    ).optimize()
    print(opt)
    print("NVT")
    md = RunMD(calculator=calculator, jarvis_atoms=opt, nsteps=5).run()
    print("NPT")
    md = RunMD(
        calculator=calculator, ensemble="npt", jarvis_atoms=opt, nsteps=5
    ).run()
"""
