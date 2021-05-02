# from alignn.train_props import train_prop_model
import os
from jarvis.tasks.queue_jobs import Queue

# d=data('dft_3d')

props = [
    "e_form",
    "gap pbe",
    "mu_b",
    "bulk modulus",
    "shear modulus",
    "elastic anisotropy",
]
cwd_home = os.getcwd()
for i in props:
    model_name = "mp_" + i + "_alignn"
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    os.chdir(model_name)
    f = open("train.py", "w")
    tmp = (
        "from alignn.train_props import "
        + 'train_prop_model \ntrain_prop_model(prop="'
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
        walltime="50:00:00",
        pre_job_lines="#SBATCH --gres=gpu:1\n"
        # pre_job_lines='#SBATCH --gres=gpu:1\n. ~/.bashrc \n'
    )
    os.chdir(cwd_home)
