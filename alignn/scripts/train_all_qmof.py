"""Module to train QMOF database."""
# from alignn.train_props import train_prop_model
import os

from jarvis.tasks.queue_jobs import Queue

# d=data('dft_3d')

props = [
    "bandgap",
    "energy_total",
    "net_magmom",
    # "e_hull",
    # "bulk modulus",
    # "shear modulus",
    # "elastic anisotropy",
]
cwd_home = os.getcwd()
for i in props:
    model_name = "qmof_" + i + "_alignn"
    model_name = model_name.replace(" ", "")
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    os.chdir(model_name)
    f = open("train.py", "w")
    tmp = (
        "from alignn.train_props import "
        + "train_prop_model \ntrain_prop_model(batch_size=32"
        + ',name="alignn",dataset="qmof",id_tag="id",prop="'
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
        cores=None,
        submit_cmd=submit_cmd,
        memory="84G",
        filename=model_name,
        queue="general,singlegpu,batch,interactive",
        walltime="72:00:00",
        pre_job_lines="#SBATCH --gres=gpu:1\n"
        # pre_job_lines='#SBATCH --gres=gpu:1\n. ~/.bashrc \n'
    )
    os.chdir(cwd_home)
