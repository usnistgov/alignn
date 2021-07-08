"""Module to train JARVIS-DFT DB properties."""
# from alignn.train_props import train_prop_model
import os
from jarvis.tasks.queue_jobs import Queue

# d=data('dft_3d')

props = [
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "optb88vdw_total_energy",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "magmom_oszicar",
    "epsx",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    # "ncond",
    # "pcond",
    # "nkappa",
    # "pkappa",
    "ehull",
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
]
cwd_home = os.getcwd()
for i in props:
    model_name = "jv_" + i + "_alignn"
    os.makedirs(model_name)
    os.chdir(model_name)
    f = open("train.py", "w")
    tmp = (
        "from alignn.train_props import "
        + 'train_prop_model \ntrain_prop_model(learning_rate=0.001,prop="'
    )
    line = tmp + i + '")\n'
    f.write(line)
    f.close()
    f = open(model_name, "w")
    job_line = ". ~/.bashrc \nconda activate version \n" + "python train.py\n"
    submit_cmd = ["sbatch", model_name]
    directory = os.getcwd()
    Queue.slurm(
        job_line=job_line,
        jobname=model_name,
        directory=directory,
        submit_cmd=submit_cmd,
        memory="30G",
        filename=model_name,
        queue="singlegpu,batch,interactive",
        walltime="72:00:00",
        pre_job_lines="#SBATCH --gres=gpu:1\n"
        # pre_job_lines='#SBATCH --gres=gpu:1\n. ~/.bashrc \n'
    )
    os.chdir(cwd_home)
