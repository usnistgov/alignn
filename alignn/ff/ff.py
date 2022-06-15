"""Module for running ALIGNN-FF."""
from ase.md import MDLogger
from jarvis.core.atoms import Atoms as JarvisAtoms
import os
from ase.md.nvtberendsen import NVTBerendsen
from ase.io import Trajectory
from jarvis.db.figshare import get_jid_data
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
import numpy as np
import torch
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
from alignn.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.defects.surface import Surface
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)

# from ase.optimize.optimize import Optimizer
# from ase.io import Trajectory
# from ase.neighborlist import NeighborList
# from ase import Atoms


def default_path():
    dpath = os.path.abspath(str(os.path.join(os.path.dirname(__file__), ".")))
    print("dpath", dpath)
    return dpath


# print("default_model_path", default_model_path)


def ase_to_atoms(ase_atoms):
    """Convert ASE Atoms to JARVIS."""
    return JarvisAtoms(
        lattice_mat=ase_atoms.get_cell(),
        elements=ase_atoms.get_chemical_symbols(),
        coords=ase_atoms.get_positions(),
        #         pbc=True,
        cartesian=True,
    )


ignore_bad_restart_file = ase.calculators.calculator.Calculator._deprecated


class AlignnAtomwiseCalculator(ase.calculators.calculator.Calculator):
    """Module for ASE Calculator interface."""

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=ignore_bad_restart_file,
        label=None,
        include_stress=True,
        atoms=None,
        directory=".",
        device="cpu",
        path=".",
        model_filename="best_model.pt",
        config_filename="config.json",
        keep_data_order=False,
        classification_threshold=None,
        batch_size=None,
        epochs=None,
        output_dir=None,
        **kwargs,
    ):
        """Initialize class."""
        super(AlignnAtomwiseCalculator, self).__init__(
            restart, ignore_bad_restart_file, label, atoms, directory, **kwargs
        )

        self.device = device
        self.include_stress = include_stress

        config = loadjson(os.path.join(path, config_filename))
        config = TrainingConfig(**config)
        if type(config) is dict:
            try:
                config = TrainingConfig(**config)
            except Exception as exp:
                print("Check", exp)
        if self.include_stress:
            self.implemented_properties = ["energy", "forces", "stress"]
            if config.model.stresswise_weight == 0:
                config.model.stresswise_weight = 0.1
        else:
            self.implemented_properties = ["energy", "forces"]

        config.keep_data_order = keep_data_order
        if classification_threshold is not None:
            config.classification_threshold = float(classification_threshold)
        if output_dir is not None:
            config.output_dir = output_dir
        if batch_size is not None:
            config.batch_size = int(batch_size)
        if epochs is not None:
            config.epochs = int(epochs)

        config.model.output_features = 1

        model = ALIGNNAtomWise(config.model)
        model.state_dict()
        model.load_state_dict(
            torch.load(os.path.join(path, model_filename), map_location=device)
        )
        model.to(device)
        model.eval()

        self.net = model
        self.net.to(self.device)

    def calculate(self, atoms, properties=None, system_changes=None):
        """Calculate properties."""
        j_atoms = ase_to_atoms(atoms)
        num_atoms = j_atoms.num_atoms
        g, lg = Graph.atom_dgl_multigraph(j_atoms)
        result = self.net((g.to(self.device), lg.to(self.device)))
        # print ('stress',result["stress"].detach().numpy())
        self.results = {
            "energy": result["out"].detach().cpu().numpy() * num_atoms,
            "forces": result["grad"].detach().cpu().numpy(),
            "stress": full_3x3_to_voigt_6_stress(
                result["stress"].detach().cpu().numpy()
            )
            / 160.21766208,
            "dipole": np.zeros(3),
            "charges": np.zeros(len(atoms)),
            "magmom": 0.0,
            "magmoms": np.zeros(len(atoms)),
        }


class ForceField(object):
    """Module to run ALIGNN-FF."""

    def __init__(
        self,
        jarvis_atoms=None,
        model_path="out",
        model_filename="best_model.pt",
        include_stress=True,
        timestep=None,
        print_format=None,
        logger=None,
        logfile="alignn_ff.log",
        dyn=None,
        communicator=None,
    ):
        """Intialize class."""
        self.jarvis_atoms = jarvis_atoms
        self.model_path = model_path
        self.model_filename = model_filename
        self.include_stress = include_stress
        self.timestep = timestep
        self.atoms = self.jarvis_atoms.ase_converter()
        self.print_format = print_format
        self.dyn = dyn
        self.communicator = communicator
        self.logger = logger
        if self.timestep is None:
            self.timestep = 0.5 * units.fs
        self.logfile = logfile
        if self.print_format is None:
            self.print_format = self.example_print
        if self.logger is None:
            self.logger = MDLogger(
                self.dyn,
                self.atoms,
                self.logfile,
                stress=False,
                peratom=False,
                header=True,
                mode="a",
            )
        # print ('STRUCTURE PROVIDED:')
        # print (ase_to_atoms(self.atoms))
        # print ()
        self.atoms.set_calculator(
            AlignnAtomwiseCalculator(
                path=self.model_path,
                include_stress=self.include_stress,
                model_filename=self.model_filename,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

    def example_print(self):
        """Print info."""
        if isinstance(self.atoms, ExpCellFilter):
            self.atoms = self.atoms.atoms
        line = ""
        try:
            line = f"time={self.dyn.get_time() / units.fs: 5.0f} fs "
        except Exception:
            pass
        line += (
            f"a={self.atoms.get_cell()[0,0]: 3.3f} Ang "
            + f"b={self.atoms.get_cell()[1,1]: 3.3f} Ang "
            + f"c={self.atoms.get_cell()[2,2]: 3.3f} Ang "
            + f"Volume={self.atoms.get_volume(): 3.3f} amu/a3 "
            + f"PE={self.atoms.get_potential_energy(): 5.5f} eV "
            + f"KE={self.atoms.get_kinetic_energy(): 5.5f} eV "
            + f"T={self.atoms.get_temperature(): 3.3f} K "
            # + f" P={atoms.
            # get_isotropic_pressure(atoms.get_stress()): 5.3f} bar "
        )
        print(line)

    def set_momentum_maxwell_boltzmann(self, temperature_K=10):
        """Set initial temperature."""
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K)

    def unrelaxed_atoms(self):
        """Get energy of a system."""
        pe = self.atoms.get_potential_energy()
        # ke = self.atoms.get_kinetic_energy()
        # print("pe", pe)
        return pe

    def optimize_atoms(
        self,
        optimizer="FIRE",
        trajectory="opt.traj",
        logfile="opt.log",
        steps=1000,
        fmax=0.05,
        optimize_lattice=True,
        interval=1,
    ):
        """Optimize structure."""
        available_ase_optimizers = {
            "BFGS": BFGS,
            "LBFGS": LBFGS,
            "LBFGSLineSearch": LBFGSLineSearch,
            "FIRE": FIRE,
            "MDMin": MDMin,
            "GPMin": GPMin,
            "SciPyFminCG": SciPyFminCG,
            "SciPyFminBFGS": SciPyFminBFGS,
            "BFGSLineSearch": BFGSLineSearch,
        }

        optimizer = available_ase_optimizers.get(optimizer, None)
        if optimizer is None:
            raise ValueError("Check optimizer", optimizer)
        if optimize_lattice:
            self.atoms = ExpCellFilter(self.atoms)
        print("OPTIMIZATION")
        self.dyn = optimizer(
            self.atoms, trajectory="opt.traj", logfile="opt.log"
        )
        self.dyn.attach(self.print_format, interval=interval)
        self.dyn.run(fmax=fmax, steps=steps)
        return ase_to_atoms(self.atoms)

    def run_nve_velocity_verlet(
        self,
        filename="ase_nve",
        interval=1,
        steps=1000,
    ):
        """Run NVE."""
        print("NVE VELOCITY VERLET")
        self.dyn = VelocityVerlet(self.atoms, self.timestep)
        # Create monitors for logfile and a trajectory file
        # logfile = os.path.join(".", "%s.log" % filename)
        trajfile = os.path.join(".", "%s.traj" % filename)
        trajectory = Trajectory(trajfile, "w", self.atoms)
        # Attach monitors to trajectory
        self.dyn.attach(self.logger, interval=interval)
        self.dyn.attach(self.print_format, interval=interval)
        self.dyn.attach(trajectory.write, interval=interval)
        self.dyn.run(steps)
        return ase_to_atoms(self.atoms)

    def run_nvt_langevin(
        self,
        filename="ase_nvt_langevin",
        interval=1,
        temperature_K=300,
        steps=1000,
        friction=1e-4,
    ):
        """Run NVT."""
        print("NVT LANGEVIN")
        self.dyn = Langevin(
            self.atoms,
            self.timestep,
            temperature_K=temperature_K,
            friction=friction,
            communicator=self.communicator,
        )
        # Create monitors for logfile and a trajectory file
        # logfile = os.path.join(".", "%s.log" % filename)
        trajfile = os.path.join(".", "%s.traj" % filename)
        trajectory = Trajectory(trajfile, "w", self.atoms)
        # Attach monitors to trajectory
        self.dyn.attach(self.logger, interval=interval)
        self.dyn.attach(self.print_format, interval=interval)
        self.dyn.attach(trajectory.write, interval=interval)
        self.dyn.run(steps)
        return ase_to_atoms(self.atoms)

    def run_nvt_andersen(
        self,
        filename="ase_nvt_andersen",
        interval=1,
        temperature_K=300,
        steps=1000,
        andersen_prob=1e-1,
    ):
        """Run NVT."""
        print("NVT ANDERSEN")
        self.dyn = Andersen(
            self.atoms,
            self.timestep,
            temperature_K=temperature_K,
            andersen_prob=andersen_prob,
            communicator=self.communicator,
        )
        # Create monitors for logfile and a trajectory file
        # logfile = os.path.join(".", "%s.log" % filename)
        trajfile = os.path.join(".", "%s.traj" % filename)
        trajectory = Trajectory(trajfile, "w", self.atoms)
        # Attach monitors to trajectory
        self.dyn.attach(self.logger, interval=interval)
        self.dyn.attach(self.print_format, interval=interval)
        self.dyn.attach(trajectory.write, interval=interval)
        self.dyn.run(steps)
        return ase_to_atoms(self.atoms)

    def run_nvt_berendsen(
        self,
        filename="ase_nvt_berendsen",
        interval=1,
        temperature_K=300,
        steps=1000,
        taut=None,
    ):
        """Run NVT."""
        print("NVT BERENDSEN")
        if taut is None:
            taut = 100 * self.timestep
        self.dyn = NVTBerendsen(
            self.atoms,
            self.timestep,
            temperature_K=temperature_K,
            taut=taut,
            communicator=self.communicator,
        )
        # Create monitors for logfile and a trajectory file
        # logfile = os.path.join(".", "%s.log" % filename)
        trajfile = os.path.join(".", "%s.traj" % filename)
        trajectory = Trajectory(trajfile, "w", self.atoms)
        # Attach monitors to trajectory
        self.dyn.attach(self.logger, interval=interval)
        self.dyn.attach(self.print_format, interval=interval)
        self.dyn.attach(trajectory.write, interval=interval)
        self.dyn.run(steps)
        return ase_to_atoms(self.atoms)

    def run_npt_nose_hoover(
        self,
        filename="ase_npt_nose_hoover",
        interval=1,
        temperature_K=300,
        steps=1000,
        externalstress=0.0,
        taut=None,
    ):
        """Run NPT."""
        print("NPT: Combined Nose-Hoover and Parrinello-Rahman dynamics")
        if taut is None:
            taut = 100 * self.timestep
        self.dyn = NPT(
            self.atoms,
            self.timestep,
            temperature_K=temperature_K,
            externalstress=externalstress,
        )
        # Create monitors for logfile and a trajectory file
        # logfile = os.path.join(".", "%s.log" % filename)
        trajfile = os.path.join(".", "%s.traj" % filename)
        trajectory = Trajectory(trajfile, "w", self.atoms)
        # Attach monitors to trajectory
        self.dyn.attach(self.logger, interval=interval)
        self.dyn.attach(self.print_format, interval=interval)
        self.dyn.attach(trajectory.write, interval=interval)
        self.dyn.run(steps)
        return ase_to_atoms(self.atoms)


def ev_curve(
    atoms=None,
    dx=np.arange(-0.05, 0.05, 0.01),
    model_path=".",
    model_filename="best_model.pt",
    fig_name="eos.png",
    on_relaxed_struct=False,
):
    """Get EV curve."""
    if on_relaxed_struct:
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            model_filename=model_filename,
        )
        relaxed = ff.optimize_atoms()
    else:
        relaxed = atoms
    y = []
    vol = []
    for i in dx:
        s1 = relaxed.strain_atoms(i)
        ff = ForceField(
            jarvis_atoms=s1,
            model_path=model_path,
            model_filename=model_filename,
        )
        energy = ff.unrelaxed_atoms()
        y.append(energy)
        vol.append(s1.volume)
    x = np.array(dx)
    y = np.array(y)
    eos = EquationOfState(vol, y, eos="murnaghan")
    v0, e0, B = eos.fit()
    kv = B / kJ * 1.0e24  # , 'GPa')
    print("KV", kv)
    # eos.plot(show=True)
    eos.plot(fig_name)
    return x, y, eos, kv


def vacancy_formation(
    atoms=None,
    jid="",
    dataset="dft_3d",
    on_conventional_cell=False,
    enforce_c_size=8,
    extend=1,
    model_path=".",
    model_filename="best_model.pt",
    using_wyckoffs=True,
    on_relaxed_struct=False,
):
    """Get vacancy energy."""
    if atoms is None:
        from jarvis.db.figshare import data

        dft_3d = data(dataset)
        for i in dft_3d:
            if i["jid"] == jid:
                atoms = JarvisAtoms.from_dict(i["atoms"])
    if on_relaxed_struct:
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            model_filename=model_filename,
        )
        relaxed = ff.optimize_atoms()
    else:
        relaxed = atoms

    strts = Vacancy(relaxed).generate_defects(
        on_conventional_cell=on_conventional_cell,
        enforce_c_size=enforce_c_size,
        extend=extend,
    )
    mem = []
    for j in strts:
        strt = JarvisAtoms.from_dict(j.to_dict()["defect_structure"])
        name = (
            str(jid)
            + "_"
            + str(strt.composition.reduced_formula)
            + "_"
            + j.to_dict()["symbol"]
            + "_"
            + j.to_dict()["wyckoff_multiplicity"]
        )
        print(name)
        bulk_atoms = JarvisAtoms.from_dict(j.to_dict()["atoms"])
        ff = ForceField(
            jarvis_atoms=bulk_atoms,
            model_path=model_path,
            model_filename=model_filename,
        )
        energy = ff.unrelaxed_atoms()
        # Bulk EPA
        pred_bulk_energy = energy / bulk_atoms.num_atoms
        defective_atoms = strt
        # print ('defective_atoms',strt)
        # defective_energy = i["defective_energy"]
        # pred_def_energy = (
        #    atom_to_energy(atoms=defective_atoms, only_energy=only_energy)
        #    * defective_atoms.num_atoms
        # )
        ff = ForceField(
            jarvis_atoms=defective_atoms,
            model_path=model_path,
            model_filename=model_filename,
        )
        pred_def_energy = ff.unrelaxed_atoms()

        chem_pot = unary_energy(j.to_dict()["symbol"].replace(" ", ""))
        # print('pred_def_energy',pred_def_energy)
        symb = j.to_dict()["symbol"].replace(" ", "")
        # print('chem_pot',symb,unary_energy(symb))
        Ef = (
            pred_def_energy
            - (defective_atoms.num_atoms + 1) * pred_bulk_energy
            + chem_pot
        )
        # print (name,Ef2,j.to_dict()["symbol"])
        info = {}
        info["jid"] = jid
        info["symb"] = symb
        info["E_vac"] = Ef
        info["wyckoff"] = j.to_dict()["wyckoff_multiplicity"]
        mem.append(info)
    return mem


def surface_energy(
    atoms=None,
    max_index=1,
    on_relaxed_struct=True,
    model_path=".",
    model_filename="best_model.pt",
):
    """Get surface energy."""
    if on_relaxed_struct:
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            model_filename=model_filename,
        )
        atoms = ff.optimize_atoms()

    atoms_cvn = Spacegroup3D(atoms).conventional_standard_structure
    # energy = atom_to_energy(atoms=atoms_cvn, only_energy=only_energy)

    indices = symmetrically_distinct_miller_indices(
        max_index=max_index, cvn_atoms=atoms_cvn
    )
    # indices = [[0, 0, 1]]
    ff = ForceField(
        jarvis_atoms=atoms_cvn,
        model_path=model_path,
        model_filename=model_filename,
    )
    epa = ff.unrelaxed_atoms() / atoms_cvn.num_atoms
    # epa = energy  # / atoms_cvn.num_atoms
    mem = []
    for j in indices:
        strt = Surface(atoms=atoms_cvn, indices=j).make_surface()

        name = (
            str(strt.composition.reduced_formula)
            + "_"
            + str("_".join(map(str, j)))
        )

        # info_def = get_energy(strt, relax_atoms=False)
        ff = ForceField(
            jarvis_atoms=strt,
            model_path=model_path,
            model_filename=model_filename,
        )
        energy = ff.unrelaxed_atoms()  # / atoms_cvn.num_atoms

        m = np.array(strt.lattice_mat)
        surf_area = np.linalg.norm(np.cross(m[0], m[1]))
        surf_en = 16 * (energy - epa * (strt.num_atoms)) / (2 * surf_area)

        # print("Surface name", name, surf_en)
        info = {}
        info["name"] = name
        info["surf_en"] = surf_en

        mem.append(info)

        # return name, def_en,epa
    return mem


if __name__ == "__main__":

    atoms = Spacegroup3D(
        JarvisAtoms.from_dict(
            get_jid_data(jid="JVASP-816", dataset="dft_3d")["atoms"]
        )
    ).conventional_standard_structure
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
    # xx = ff.run_npt_nose_hoover(steps=5)
    # print(xx)