"""Train multiple properties using CGCNN."""
# For comparison purposes only
import os

from jarvis.tasks.queue_jobs import Queue

props = [
    "avg_elec_mass",
    "avg_hole_mass",
    "max_efg",
    "magmom_oszicar",
    "epsx",
    "kpoint_length_unit",
    "encut",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
    # "ncond",
    # "pcond",
    # "nkappa",
    # "pkappa",
    # "density",
    # "max_ir_mode",
    # "min_ir_mode",
]
dataset = "dft_3d"
id_tag = "jid"
cwd_home = os.getcwd()
for i in props:
    if not os.path.exists(i):
        os.makedirs(i)
    os.chdir(i)
    f = open("train.py", "w")
    tmp = (
        "from alignn.scripts.train_cgcnn_repo import cgcnn_pred\n"
        + 'cgcnn_pred(prop="'
        + i
        + '",dataset_name="'
        + dataset
        + '",id_tag="'
        + id_tag
        + '")\n'
    )
    f.write(tmp)
    f.close()
    f = open(i, "w")
    job_line = ". ~/.bashrc \nconda activate cgcnn \n" + "python train.py\n"
    submit_cmd = ["sbatch", i]
    directory = os.getcwd()

    Queue.slurm(
        job_line=job_line,
        jobname=i,
        directory=directory,
        submit_cmd=submit_cmd,
        memory="90G",
        filename=i,
        cores=None,
        queue="interactive",
        walltime="8:00:00",
        pre_job_lines="#SBATCH --gres=gpu:1\n"
        # pre_job_lines='#SBATCH --gres=gpu:1\n. ~/.bashrc \n'
    )
    os.chdir(cwd_home)
