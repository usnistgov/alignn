"""Module to train JARVIS-DFT DB properties for classification."""
# from alignn.train_props import train_prop_model
import os
from jarvis.tasks.queue_jobs import Queue

# d=data('dft_3d')

props = [
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "slme",
    "ehull",
    "magmom_oszicar",
    "spillage",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
]
thresholds = {
    "optb88vdw_bandgap": 0.01,
    "mbj_bandgap": 0.01,
    "slme": 10,
    "ehull": 0.1,
    "magmom_oszicar": 0.05,
    "spillage": 0.1,
    "n-Seebeck": -100,
    "p-Seebeck": 100,
    "n-powerfact": 1000,
    "p-powerfact": 1000,
}
cwd_home = os.getcwd()
for i in props:
    model_name = "jv_" + i + "_alignn_class"
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    os.chdir(model_name)
    f = open("train.py", "w")
    tmp = (
        "from alignn.train_props import "
        + "train_prop_model \ntrain_prop_model(batch_size=64,"
        + 'learning_rate=0.001,n_epochs=100,prop="'
    )
    line = (
        tmp
        + i
        + '"'
        + ",classification_threshold="
        + str(thresholds[i])
        + ")\n"
    )
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
        memory="34G",
        filename=model_name,
        queue="singlegpu,batch",
        walltime="72:00:00",
        pre_job_lines="#SBATCH --gres=gpu:1\n"
        # pre_job_lines='#SBATCH --gres=gpu:1\n. ~/.bashrc \n'
    )
    os.chdir(cwd_home)
