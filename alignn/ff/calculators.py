"""Module for running ALIGNN-FF."""

import matplotlib.pyplot as plt
from jarvis.core.atoms import Atoms as JarvisAtoms
import os
import requests
import ase.calculators.calculator
from ase.stress import full_3x3_to_voigt_6_stress
from jarvis.db.jsonutils import loadjson
from alignn.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.models.ealignn_atomwise import (
    eALIGNNAtomWise,
    eALIGNNAtomWiseConfig,
)
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import matplotlib.pyplot as plt  # noqa
import zipfile
import numpy as np
from tqdm import tqdm
import torch

# Reference: https://doi.org/10.1039/D2DD00096B


def get_all_models():
    json_path = os.path.join(
        os.path.dirname(__file__), "all_models_alignn_atomwise.json"
    )
    return loadjson(json_path)


def get_all_models_prop():
    json_path = os.path.join(
        os.path.dirname(__file__), "all_models_alignn.json"
    )
    return loadjson(json_path)


def get_figshare_model_ff(
    model_name="v5.27.2024", dir_path=None, filename="best_model.pt"
):
    """Get ALIGNN-FF torch models from figshare."""
    all_models_ff = get_all_models()
    # https://doi.org/10.6084/m9.figshare.23695695
    if dir_path is None:
        dir_path = str(os.path.join(os.path.dirname(__file__), model_name))
    # cwd=os.getcwd()
    dir_path = os.path.abspath(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # os.chdir(dir_path)
    url = all_models_ff[model_name]
    zfile = model_name + ".zip"
    path = str(os.path.join(dir_path, zfile))
    # path = str(os.path.join(os.path.dirname(__file__), zfile))
    print("dir_path", dir_path)
    best_path = os.path.join(dir_path, filename)
    if not os.path.exists(best_path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        zp = zipfile.ZipFile(path)
        names = zp.namelist()
        chks = []
        cfg = []
        for i in names:
            if filename in i:
                tmp = i
                chks.append(i)
            if "config.json" in i:
                cfg = i

        config = zipfile.ZipFile(path).read(cfg)
        # print("Loading the zipfile...", zipfile.ZipFile(path).namelist())
        data = zipfile.ZipFile(path).read(tmp)

        # new_file, filename = tempfile.mkstemp()
        filename = os.path.join(dir_path, filename)
        with open(filename, "wb") as f:
            f.write(data)
        filename = os.path.join(dir_path, "config.json")
        with open(filename, "wb") as f:
            f.write(config)
        os.remove(path)
    # print("Using model file", url, "from ", chks)
    # print("Path", os.path.abspath(path))
    # print("Config", os.path.abspath(cfg))
    return dir_path


def default_path():
    """Get default model path."""
    dpath = get_figshare_model_ff(model_name="v12.2.2024_dft_3d_307k")
    # dpath = get_figshare_model_ff(model_name="v5.27.2024")
    # dpath = get_figshare_model_ff(model_name="v8.29.2024_dft_3d")
    # dpath = get_figshare_model_ff(model_name="alignnff_wt10")
    # dpath = get_figshare_model_ff(model_name="alignnff_fmult")
    # print("model_path", dpath)
    return dpath


def mp_2mill():
    """Get default model path."""
    dpath = get_figshare_model_ff(model_name="v12.2.2024_mp_1.5mill")
    # print("model_path", dpath)
    return dpath


def mp_167k():
    """Get default model path."""
    dpath = get_figshare_model_ff(model_name="v12.2.2024_mp_187k")
    # print("model_path", dpath)
    return dpath


def jv_307k():
    """Get MPtraj model path."""
    dpath = get_figshare_model_ff(model_name="v8.29.2024_mpf")
    # print("model_path", dpath)
    return dpath


def wt01_path():
    """Get defaukt model path."""
    dpath = get_figshare_model_ff(model_name="alignnff_wt01")
    # print("model_path", dpath)
    return dpath


def wt1_path():
    """Get defaukt model path."""
    dpath = get_figshare_model_ff(model_name="alignnff_wt1")
    # print("model_path", dpath)
    return dpath


def wt10_path():
    """Get defaukt model path."""
    dpath = get_figshare_model_ff(model_name="alignnff_wt10")
    # print("model_path", dpath)
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
        intensive=True,
        atoms=None,
        directory=".",
        device=None,
        model=None,
        config=None,
        path=None,
        model_filename="best_model.pt",
        config_filename="config.json",
        output_dir=None,
        batch_stress=True,
        force_mult_natoms=False,
        force_mult_batchsize=True,
        force_multiplier=1,
        stress_wt=0.05,
    ):
        """Initialize class."""
        super(AlignnAtomwiseCalculator, self).__init__(
            restart, ignore_bad_restart_file, label, atoms, directory
        )  # , **kwargs
        self.model = model
        self.device = device
        self.intensive = intensive
        self.config = config
        self.include_stress = include_stress
        self.stress_wt = stress_wt
        self.force_mult_natoms = force_mult_natoms
        self.force_mult_batchsize = force_mult_batchsize
        self.force_multiplier = force_multiplier
        self.trained_stress = False
        if path is None and model is None:
            path = default_path()
        if self.config is None:
            config = loadjson(os.path.join(path, config_filename))
            self.config = config
        if self.force_mult_natoms:
            self.config["model"]["force_mult_natoms"] = True

        if self.include_stress:
            self.implemented_properties = ["energy", "forces", "stress"]
            if (
                "stresswise_weight" in self.config["model"]
                and self.config["model"]["stresswise_weight"] == 0
            ):
                self.trained_stress = False
                self.config["model"]["stresswise_weight"] = 0.1
            else:
                self.trained_stress = True

        else:
            self.implemented_properties = ["energy", "forces"]
        if self.config["model"]["calculate_gradient"]:
            self.trained_stress = True

        if (
            batch_stress is not None
            and "atomwise" in self.config["model"]["name"]
        ):
            self.config["model"]["batch_stress"] = batch_stress
        import torch

        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if self.model is None:

            if self.config["model"]["name"] == "alignn_atomwise":
                model = ALIGNNAtomWise(
                    ALIGNNAtomWiseConfig(**self.config["model"])
                )
            elif self.config["model"]["name"] == "alignn":
                model = ALIGNN(ALIGNNConfig(**self.config["model"]))
            elif self.config["model"]["name"] == "ealignn_atomwise":
                model = eALIGNNAtomWise(
                    eALIGNNAtomWiseConfig(**self.config["model"])
                )
            model.state_dict()
            if "atomwise" in self.config["model"]["name"]:
                model.load_state_dict(
                    torch.load(
                        os.path.join(path, model_filename),
                        map_location=self.device,
                    )
                )
            else:
                model.load_state_dict(
                    torch.load(
                        os.path.join(path, model_filename),
                        map_location=self.device,
                    )["model"]
                )
            model.to(device)
            model.eval()
            self.model = model
        else:
            model = self.model

    def calculate(self, atoms, properties=None, system_changes=None):
        """Calculate properties."""
        j_atoms = ase_to_atoms(atoms)
        num_atoms = j_atoms.num_atoms
        g, lg = Graph.atom_dgl_multigraph(
            j_atoms,
            neighbor_strategy=self.config["neighbor_strategy"],
            cutoff=self.config["cutoff"],
            max_neighbors=self.config["max_neighbors"],
            atom_features=self.config["atom_features"],
            use_canonize=self.config["use_canonize"],
        )

        if self.config["model"]["alignn_layers"] > 0:
            result = self.model(
                (
                    g.to(self.device),
                    lg.to(self.device),
                    torch.tensor(atoms.cell)
                    .type(torch.get_default_dtype())
                    .to(self.device),
                )
            )
        else:
            result = self.model(
                (g.to(self.device), torch.tensor(atoms.cell).to(self.device))
            )
        # print("result",result)
        if "atomwise" in self.config["model"]["name"]:
            forces = forces = (
                result["grad"].detach().cpu().numpy() * self.force_multiplier
            )
        else:
            forces = np.zeros((3, 3))
        # print("self.trained_stress",self.trained_stress)
        if self.trained_stress:
            # if "atomwise" in self.config["model"]["name"] and self.trained_stress:
            stress = (
                full_3x3_to_voigt_6_stress(
                    result["stresses"][:3].reshape(3, 3).detach().cpu().numpy()
                )
                * self.stress_wt
                / 160.21766208
            )
        else:
            stress = np.zeros((3, 3))
        # stress = (
        #    full_3x3_to_voigt_6_stress(
        #        result["stresses"][:3].reshape(3, 3).detach().cpu().numpy()
        #    )
        #    * self.stress_wt
        #    / 160.21766208
        # )
        if "atomwise" in self.config["model"]["name"]:
            energy = result["out"].detach().cpu().numpy()
        else:
            energy = result.detach().cpu().numpy()
        if self.intensive:
            energy *= num_atoms
        if self.force_mult_natoms:
            forces *= num_atoms
        if self.force_mult_batchsize:
            forces *= self.config["batch_size"]

        # print("stress cal",stress)
        self.results = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
        }


class iAlignnAtomwiseCalculator(ase.calculators.calculator.Calculator):
    """Module for ASE Calculator interface."""

    def __init__(
        self,
        atoms=None,
        device=None,
        prop_path=None,
        prop_model=None,
        prop_config=None,
        prop_model_filename="best_model.pt",
        prop_config_filename="config.json",
        ff_path=None,
        ff_model=None,
        ff_model_filename="best_model.pt",
        ff_config_filename="config.json",
        ff_config=None,
        stress_wt=0.05,
        props=[
            "cbm",
            "vbm",
            "gap",
            "efermi",
            "optb88vdw_bandgap",
            "mbj_bandgap",
            "spillage",
            "slme",
            "bulk_modulus_kv",
            "shear_modulus_gv",
            "n-Seebeck",
            "n-powerfact",
            "avg_elec_mass",
            "avg_hole_mass",
            "epsx",
            "mepsx",
            "max_efg",
            "dfpt_piezo_max_dielectric",
            "dfpt_piezo_max_dij",
            "exfoliation_energy",
            "Tc_supercon",
            "magmom_oszicar",
        ],
    ):
        """Initialize class."""
        super().__init__()
        # super().__init__(**kwargs)
        self.device = device
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.stress_wt = stress_wt
        self.trained_stress = False
        self.props = props
        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
            "charges",
            "magmoms",
        ] + self.props
        if ff_path is None and ff_model is None:
            ff_path = get_figshare_model_ff(
                model_name="v12.2.2024_dft_3d_307k"
            )
        if ff_config is None:
            ff_config = loadjson(os.path.join(ff_path, ff_config_filename))
            ff_config["model"]["stresswise_weight"] = 0.1
        ff_model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**ff_config["model"]))
        ff_model.load_state_dict(
            torch.load(
                os.path.join(ff_path, ff_model_filename),
                map_location=self.device,
            )
        )
        ff_model.eval()
        self.ff_model = ff_model
        if prop_path is None and prop_model is None:
            prop_path = get_figshare_model_ff(
                model_name="v2024.12.12_dft_3d_multi_prop"
            )
        if prop_config is None:
            prop_config = loadjson(
                os.path.join(prop_path, prop_config_filename)
            )
        self.prop_config = prop_config
        self.ff_config = ff_config
        prop_model = ALIGNNAtomWise(
            ALIGNNAtomWiseConfig(**prop_config["model"])
        )
        prop_model.load_state_dict(
            torch.load(
                os.path.join(prop_path, prop_model_filename),
                map_location=self.device,
            )
        )
        prop_model.eval()
        self.prop_model = prop_model

    def calculate(
        self,
        atoms,
        properties=None,
        system_changes=None,
    ):
        # def calculate(self, atoms, properties=None, system_changes=None):
        """Calculate properties."""
        j_atoms = ase_to_atoms(atoms)
        num_atoms = j_atoms.num_atoms
        g, lg = Graph.atom_dgl_multigraph(
            j_atoms,
            neighbor_strategy=self.ff_config["neighbor_strategy"],
            cutoff=self.ff_config["cutoff"],
            max_neighbors=self.ff_config["max_neighbors"],
            atom_features=self.ff_config["atom_features"],
            use_canonize=self.ff_config["use_canonize"],
        )
        # print("config",self.ff_config)
        result_ff = self.ff_model(
            (
                g.to(self.device),
                lg.to(self.device),
                torch.tensor(atoms.cell)
                .type(torch.get_default_dtype())
                .to(self.device),
            )
        )
        forces = forces = result_ff["grad"].detach().cpu().numpy()

        stress = (
            full_3x3_to_voigt_6_stress(
                result_ff["stresses"][:3].reshape(3, 3).detach().cpu().numpy()
            )
            * self.stress_wt
            / 160.21766208
        )
        energy = result_ff["out"].detach().cpu().numpy()

        result_prop = self.prop_model(
            (
                g.to(self.device),
                lg.to(self.device),
                torch.tensor(atoms.cell)
                .type(torch.get_default_dtype())
                .to(self.device),
            )
        )
        atomwise = result_prop["atomwise_pred"].detach().cpu().numpy().tolist()
        additional = (
            result_prop["additional"].detach().cpu().numpy().tolist()[0]
        )
        charges = [i[0] for i in atomwise]
        magmoms = [i[1] for i in atomwise]

        # print('additional',len(additional),len(self.props))
        results = {
            "energy": energy * num_atoms,
            "forces": forces,
            "stress": stress,
            "charges": charges,
            "magmoms": magmoms,
        }
        for i, j in zip(self.props, additional):
            results[i] = j
            if "gap" in i and j < 0:
                results[i] = 0
        self.results = results
        # print(self.results)
