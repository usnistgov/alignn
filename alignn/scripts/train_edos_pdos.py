"""Module to train electron/phonon DOS."""
import os
from jarvis.tasks.queue_jobs import Queue


props = [
    "edos_up",
    "pdos_elast",
]
cwd_home = os.getcwd()
for i in props:
    model_name = "edospdos_" + i + "_alignnn"
    model_name = model_name.replace(" ", "")
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    os.chdir(model_name)
    f = open("train.py", "w")
    tmp = (
        "from alignn.train_props import "
        + "train_prop_model \ntrain_prop_model"
        + '(batch_size=64,name="alignn",dataset="edos_pdos",prop="'
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
        memory="34G",
        filename=model_name,
        queue="general,singlegpu,batch,interactive",
        walltime="7:00:00",
        pre_job_lines="#SBATCH --gres=gpu:1\n"
        # pre_job_lines='#SBATCH --gres=gpu:1\n. ~/.bashrc \n'
    )
    os.chdir(cwd_home)
