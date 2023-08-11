#!/usr/bin/env python

"""Module to run ALIGNN-FF."""
import argparse
import sys
from jarvis.core.atoms import Atoms
import re

# from jarvis.core.graphs import Graph
from alignn.ff.ff import (
    default_path,
    ev_curve,
    surface_energy,
    vacancy_formation,
    ForceField,
    get_interface_energy,
)
import numpy as np

parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network Force-field"
)
parser.add_argument(
    "--model_path",
    default="na",
    help="Provide model path, id na, default path chosen ",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument("--temperature_K", default="300", help="Temperature in K")
parser.add_argument(
    "--initial_temperature_K", default="0.1", help="Intial Temperature in K"
)
parser.add_argument("--timestep", default="0.5", help="Timestep in fs")
parser.add_argument("--on_relaxed_struct", default="No", help="Yes/No.")
parser.add_argument(
    "--file_path",
    default="alignn/examples/sample_data/POSCAR-JVASP-10.vasp",
    help="Path to file.",
)

parser.add_argument(
    "--task",
    default="optimize",
    help="Select task for ALIGNN-FF"
    + " such as unrelaxed_energy/optimze/nvt_lagevin/nve_velocity_verlet/npt/"
    + "npt_berendsen/nvt_berendsen/ev_curve/vacancy_energy/surface_energy/"
    + "interface",
)

parser.add_argument("--md_steps", default=100, help="Provide md steps.")
intf_line = (
    "Provide POSCAR_for_film POSCAR_for_subs "
    + "miller_index_film(e.g.111) "
    + "miller_index_subs(e.g.001) "
    + "film_thickness(e.g.25)"
    + " subs_thickness(e.g.25) separation(e.g.3.0)"
    + "e.g. POSCAR-film.vasp POSCAR-subs.vasp 111 001 25.0 25.0 3.0"
)
parser.add_argument("--interface_info", default=None, help=intf_line)

parser.add_argument(
    "--device",
    default=None,
    help="set device for executing the model [e.g. cpu, cuda, cuda:2]"
)

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    model_path = args.model_path
    file_path = args.file_path
    file_format = args.file_format
    task = args.task
    temperature_K = float(args.temperature_K)
    initial_temperature_K = float(args.initial_temperature_K)
    on_relaxed_struct_yn = args.on_relaxed_struct
    timestep = float(args.timestep)
    if on_relaxed_struct_yn.lower() == "yes":
        on_relaxed_struct = True
    else:
        on_relaxed_struct = False

    steps = int(args.md_steps)
    if file_format == "poscar":
        atoms = Atoms.from_poscar(file_path)
    elif file_format == "cif":
        atoms = Atoms.from_cif(file_path)
    elif file_format == "xyz":
        atoms = Atoms.from_xyz(file_path, box_size=500)
    elif file_format == "pdb":
        atoms = Atoms.from_pdb(file_path, max_lat=500)
    else:
        raise NotImplementedError("File format not implemented", file_format)
    if model_path == "na":
        model_path = default_path()

    if task == "unrelaxed_energy":
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        energy = ff.unrelaxed_atoms()
        print("Energy(eV)", energy)
    if task == "optimize":
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        opt, en, fs = ff.optimize_atoms()
        print("initial struct:")
        print(atoms)
        print("final struct:")
        print(opt)
    if task == "nvt_berendsen":
        print("initial struct:")
        print(atoms)
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            timestep=timestep,
        )
        nptt = ff.run_nvt_berendsen(
            steps=steps,
            temperature_K=temperature_K,
            initial_temperature_K=initial_temperature_K,
        )
        print("final struct:")
        print(nptt)
    if task == "ev_curve":
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        ev = ev_curve(
            atoms=atoms,
            model_path=model_path,
            on_relaxed_struct=on_relaxed_struct,
        )
    if task == "vacancy_energy":
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        vac = vacancy_formation(
            atoms=atoms,
            model_path=model_path,
            on_relaxed_struct=on_relaxed_struct,
        )
        print(vac)
    if task == "surface_energy":
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        surf = surface_energy(
            atoms=atoms,
            model_path=model_path,
            on_relaxed_struct=on_relaxed_struct,
        )
        print(surf)
    if task == "interface":
        interface_info = args.interface_info.split()
        film_atoms = Atoms.from_poscar(interface_info[0])
        subs_atoms = Atoms.from_poscar(interface_info[1])
        film_index = np.array(
            [i for i in re.split("_", interface_info[2]) if i != ""],
            # [i for i in re.split("(\d)", interface_info[2]) if i != ""],
            dtype="int",
        )
        subs_index = np.array(
            [i for i in re.split("_", interface_info[3]) if i != ""],
            # [i for i in re.split("(\d)", interface_info[3]) if i != ""],
            dtype="int",
        )
        film_thickness = float(interface_info[4])
        subs_thickness = float(interface_info[5])
        seperation = float(interface_info[6])
        intf = get_interface_energy(
            film_atoms=film_atoms,
            subs_atoms=subs_atoms,
            model_path=model_path,
            film_index=film_index,
            subs_index=subs_index,
            seperation=seperation,
            film_thickness=film_thickness,
            subs_thickness=subs_thickness,
        )
        print("Film:\n")
        print(intf["film_sl"])
        print("Substrate:\n")
        print(intf["subs_sl"])
        print("Interface:\n")
        print(intf["optimized_interface"])
        print("Interface energy(J/m2):\n")
        print(intf["interface_energy"])
    if task == "npt_berendsen":
        print("initial struct:")
        print(atoms)
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            timestep=timestep,
        )
        nptt = ff.run_npt_berendsen(
            steps=steps,
            temperature_K=temperature_K,
            initial_temperature_K=initial_temperature_K,
        )
        print("final struct:")
        print(nptt)
    if task == "nvt_lagevin":
        print("initial struct:")
        print(atoms)
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            timestep=timestep,
        )
        lang = ff.run_nvt_langevin(
            steps=steps,
            temperature_K=temperature_K,
            initial_temperature_K=initial_temperature_K,
        )
        print("final struct:")
        print(lang)

    if task == "nve_velocity_verlet":
        print("initial struct:")
        print(atoms)
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            timestep=timestep,
        )
        vv = ff.run_nve_velocity_verlet(
            steps=steps, initial_temperature_K=initial_temperature_K
        )
        print("final struct:")
        print(vv)
    if task == "npt":
        print("initial struct:")
        print(atoms)
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
            timestep=timestep,
        )
        nptt = ff.run_npt_nose_hoover(
            steps=steps,
            temperature_K=temperature_K,
            initial_temperature_K=initial_temperature_K,
        )
        print("final struct:")
        print(nptt)
        