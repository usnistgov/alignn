![alt text](https://github.com/usnistgov/alignn/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/usnistgov/alignn/branch/main/graph/badge.svg?token=S5X4OYC80V)](https://codecov.io/gh/usnistgov/alignn)
[![PyPI version](https://badge.fury.io/py/alignn.svg)](https://badge.fury.io/py/alignn)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/usnistgov/alignn)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/usnistgov/alignn)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/usnistgov/alignn)
[![Downloads](https://pepy.tech/badge/alignn)](https://pepy.tech/project/alignn)
<!--
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/formation-energy-on-materials-project)](https://paperswithcode.com/sota/formation-energy-on-materials-project?p=atomistic-line-graph-neural-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/band-gap-on-materials-project)](https://paperswithcode.com/sota/band-gap-on-materials-project?p=atomistic-line-graph-neural-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/formation-energy-on-qm9)](https://paperswithcode.com/sota/formation-energy-on-qm9?p=atomistic-line-graph-neural-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/formation-energy-on-jarvis-dft-formation)](https://paperswithcode.com/sota/formation-energy-on-jarvis-dft-formation?p=atomistic-line-graph-neural-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/band-gap-on-jarvis-dft)](https://paperswithcode.com/sota/band-gap-on-jarvis-dft?p=atomistic-line-graph-neural-network-for)
-->

# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Examples](#example)
* [Pre-trained models](#pretrained)
* [JARVIS-ALIGNN webapp](#webapp)
* [ALIGNN-FF & ASE Calculator](#alignnff)
* [Peformances on a few datasets](#performances)
* [Useful notes](#notes)
* [References](#refs)
* [How to contribute](#contrib)
* [Correspondence](#corres)
* [Funding support](#fund)

<a name="intro"></a>
# ALIGNN & ALIGNN-FF (Introduction)
The Atomistic Line Graph Neural Network (https://www.nature.com/articles/s41524-021-00650-1)  introduces a new graph convolution layer that explicitly models both two and three body interactions in atomistic systems. This is achieved by composing two edge-gated graph convolution layers, the first applied to the atomistic line graph *L(g)* (representing triplet interactions) and the second applied to the atomistic bond graph *g* (representing pair interactions).

Atomisitic line graph neural network-based FF (ALIGNN-FF) (https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00096b ) can be used to model both structurally and chemically diverse systems with any combination of 89 elements from the periodic table, specially for structural optimization. To train the ALIGNN-FF model, we have used the JARVIS-DFT dataset which contains around 75000 materials and 4 million energy-force entries, out of which 307113 are used in the training. These models can be further finetuned, or new models can be developed from scratch on a new dataset.



![ALIGNN layer schematic](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/schematic_lg.jpg)

<a name="install"></a>
Installation
-------------------------
First create a conda environment:
Install miniconda environment from https://conda.io/miniconda.html
Based on your system requirements, you'll get a file something like 'Miniconda3-latest-XYZ'.

Now,

```
bash Miniconda3-latest-Linux-x86_64.sh (for linux)
bash Miniconda3-latest-MacOSX-x86_64.sh (for Mac)
```
Download 32/64 bit python 3.10 miniconda exe and install (for windows)

#### Method 1 (conda based installation, recommended) 

Now, let's make a conda environment, say "my_alignn", choose other name as you like::
```
conda create --name my_alignn python=3.10 -y
conda activate my_alignn
conda install dgl=2.1.0 pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install alignn -y
```



#### Method 2 (GitHub based installation)

You can laso install a development version of alignn by cloning the repository and installing in place with pip:

```
conda create --name my_alignn python=3.10 -y
conda activate my_alignn
conda install dgl=2.1.0 pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
git clone https://github.com/usnistgov/alignn
cd alignn
python -m pip install -e .
```


#### Method 3 (using pypi):

As an alternate method, ALIGNN can also be installed using `pip`. Note, we have received several messages regarding dgl installation issues. You can look into dgl installation [here](https://www.dgl.ai/pages/start.html). Example for PyTorch 2.1+CUDA 12.1+Pip(Stable)+Windows:
```
pip install  -q dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
pip install alignn
```

With no GPU/CUDA:
```
pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html
pip install alignn
```

You can find out installation examples in Google Colab notebooks below

<a name="example"></a>
Examples
---------


| Notebooks                                                                                                                                      | Google&nbsp;Colab                                                                                                                                        | Descriptions                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Regression task (grpah wise prediction)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/alignn_jarvis_leaderboard.ipynb)                                                       | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/alignn_jarvis_leaderboard.ipynb)                                 | Examples for developing single output regression model for exfoliation energies of 2D materials.                                                                                                                                                                                                                                                                       |
| [Machine learning force-field training from scratch](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Train_ALIGNNFF_Mlearn.ipynb)                                                  | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Train_ALIGNNFF_Mlearn.ipynb)                            | Examples of training a machine learning force field for Silicon.                                                                                                                                                                                                                                                                                                                                 |
| [ALIGNN-FF Relaxer+EV_curve+Phonons+Interface gamma_surface+Interface separation](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Structure_Relaxation_Phonons_Interface.ipynb)                                                  | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Structure_Relaxation_Phonons_Interface.ipynb)                            | Examples of using pre-trained ALIGNN-FF force-field model.                                                                                                                                                                                                                                                                                                                                 |
| [Scaling/timing comaprison](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Timing_uMLFF.ipynb)                                                  | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Timing_uMLFF.ipynb)                            | Examples of analyzing scaling                                                                                                                                                                                                                                                                                                                                 |
| [Running MD for Melt-Quench](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Fast_Melt_Quench.ipynb)                                                  | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Fast_Melt_Quench.ipynb)                            | Examples of making amorphous structure with moelcular dynamics.                                                                                                                                                                                                                                                                                                                                 |
| [Miscellaneous tasks](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Training_ALIGNN_model_example.ipynb)                   | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Training_ALIGNN_model_example.ipynb)  | Examples for developing single output (such as formation energy, bandgaps) or multi-output (such as phonon DOS, electron DOS) Regression or Classification (such as metal vs non-metal), Using several pretrained models. |


[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg

Here, we provide examples for property prediction tasks, development of machine-learning force-fields (MLFF), usage of pre-trained property predictor, MLFFs, webapps etc.

### Dataset preparation for property prediction tasks


The main script to train model is `train_alignn.py`. A user needs at least the following info to train a model: 1) `id_prop.csv` with name of the file and corresponding value, 2) `config_example.json` a config file with training and hyperparameters.

Users can keep their structure files in `POSCAR`, `.cif`, `.xyz` or `.pdb` files in a directory. In the examples below we will use POSCAR format files. In the same directory, there should be an `id_prop.csv` file.

In this directory, `id_prop.csv`, the filenames, and correponding target values are kept in `comma separated values (csv) format`.

Here is an example of training OptB88vdw bandgaps of 50 materials from JARVIS-DFT database. The example is created using the [generate_sample_data_reg.py](https://github.com/usnistgov/alignn/blob/main/alignn/examples/sample_data/scripts/generate_sample_data_reg.py) script. Users can modify the script for more than 50 data, or make their own dataset in this format. For list of available datasets see [Databases](https://jarvis-tools.readthedocs.io/en/master/databases.html).

The dataset in split in 80:10:10 as training-validation-test set (controlled by `train_ratio, val_ratio, test_ratio`) . To change the split proportion and other parameters, change the [config_example.json](https://github.com/usnistgov/alignn/blob/main/alignn/examples/sample_data/config_example.json) file. If, users want to train on certain sets and val/test on another dataset, set `n_train`, `n_val`, `n_test` manually in the `config_example.json` and also set `keep_data_order` as True there so that random shuffle is disabled.

A brief help guide (`-h`) can be obtained as follows.

```
train_alignn.py -h
```
#### Regression example
Now, the model is trained as follows. Please increase the `batch_size` parameter to something like 32 or 64 in `config_example.json` for general trainings.

```
train_alignn.py --root_dir "alignn/examples/sample_data" --config "alignn/examples/sample_data/config_example.json" --output_dir=temp
```

#### Classification example
While the above example is for regression, the follwoing example shows a classification task for metal/non-metal based on the above bandgap values. We transform the dataset into 1 or 0 based on a threshold of 0.01 eV (controlled by the parameter, `classification_threshold`) and train a similar classification model. Currently, the script allows binary classification tasks only.
```
train_alignn.py --root_dir "alignn/examples/sample_data" --classification_threshold 0.01 --config "alignn/examples/sample_data/config_example.json" --output_dir=temp
```

#### Multi-output model example
While the above example regression was for single-output values, we can train multi-output regression models as well.
An example is given below for training formation energy per atom, bandgap and total energy per atom simulataneously. The script to generate the example data is provided in the script folder of the sample_data_multi_prop. Another example of training electron and phonon density of states is provided also.
```
train_alignn.py --root_dir "alignn/examples/sample_data_multi_prop" --config "alignn/examples/sample_data/config_example.json" --output_dir=temp
```

#### Force-field training

To train ALIGNN-FF we can use the same `train_alignn.py` script which uses `atomwise_alignn` model.

AtomWise prediction example which looks for similar setup as before but unstead of `id_prop.csv`, it requires `id_prop.json` file (see example in the sample_data_ff directory). The json contains entries such as jid, energy, forces and stress. An example to compile vasprun.xml files into a id_prop.json is kept [here](https://colab.research.google.com/gist/knc6/5513b21f5fd83a7943509ffdf5c3608b/make_id_prop.ipynb). Note ALIGNN-FF requires energy stored as energy per atom:


```
train_alignn.py --root_dir "alignn/examples/sample_data_ff" --config "alignn/examples/sample_data_ff/config_example_atomwise.json" --output_dir="temp"
```


To finetune model, use `--restart_model_path` tag as well in the above with the path of a pretrained ALIGNN-FF model with same model confurations.

```
train_alignn.py --root_dir "alignn/examples/sample_data_ff" --restart_model_path "temp/best_model.pt" --config "alignn/examples/sample_data_ff/config_example_atomwise.json" --output_dir="temp1"
```

Starting version v2024.10.30, we also allow global training for multi-output along with energy (graph wise output), forces (atomwise gradients), charges/magnetic moments etc. (atomwise but non-gradients) properties with or without additional fingerprints/features in graph. See examples [here](https://github.com/usnistgov/alignn/tree/main/alignn/examples). 

Multi-GPU training is allowed with `DistributedDataParallel` with `torchrun` command. This feature is not thoroughly tested yet. 
Example:

```
torchrun --nproc_per_node=4 train_alignn.py --root_dir DataDir --config config.json --output_dir temp
```
For multi-GPU training make sure you have correct SLURM/PBS script setup correctly such as `#SBATCH -n 4, #SBATCH -N 1, #SBATCH --gres=gpu:4` etc.

High-throughput like training: Users can also try training using multiple example scripts to run multiple dataset (such as JARVIS-DFT, Materials project, QM9_JCTC etc.). Look into the [alignn/scripts/train_*.py](https://github.com/usnistgov/alignn/tree/main/alignn/scripts) folder. This is done primarily to make the trainings more automated rather than making folder/ csv files etc.
These scripts automatically download datasets from [Databases in jarvis-tools](https://jarvis-tools.readthedocs.io/en/master/databases.html) and train several models. Make sure you specify your specific queuing system details in the scripts.


Additional example trainings for property prediction task: [2D-exfoliation energy](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/alignn_jarvis_leaderboard.ipynb), [superconductor transition temperature](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Sc.ipynb).

An example for training MLFF for Silicon is provided [here](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Train_ALIGNNFF_Mlearn.ipynb). It is highly recommeded to get familiar with this example before developing a new model. 

<a name="pretrained"></a>
Using pre-trained models
-------------------------

All the trained models are distributed on [Figshare](https://figshare.com/projects/ALIGNN_models/126478.

The [pretrained.py script](https://github.com/usnistgov/alignn/blob/develop/alignn/pretrained.py) can be applied to use them. These models can be used to directly make predictions.

A brief help section (`-h`) is shown using:

```
pretrained.py -h
```
An example of prediction formation energy per atom using JARVIS-DFT dataset trained model is shown below:

```
pretrained.py --model_name jv_formation_energy_peratom_alignn --file_format poscar --file_path alignn/examples/sample_data/POSCAR-JVASP-10.vasp
```

A pretrained ALIGNN-FF (under active development right now) can be used for predicting several properties, such as:

```
run_alignn_ff.py --file_path alignn/examples/sample_data/POSCAR-JVASP-10.vasp --task="unrelaxed_energy"
run_alignn_ff.py --file_path alignn/examples/sample_data/POSCAR-JVASP-10.vasp --task="optimize"
run_alignn_ff.py --file_path alignn/examples/sample_data/POSCAR-JVASP-10.vasp --task="ev_curve"
```

To know about other tasks, type.

```
run_alignn_ff.py -h
```

Several supporting scripts for stucture optimization, equation of states, phonon and related calculations are provided in the repo as well. If you need further assistance for a particular task, feel free to raise an GitHus issue.

<a name="webapp"></a>
Web-apps
------------

A basic web-app is for direct-prediction available at [JARVIS-ALIGNN app](https://jarvis.nist.gov/jalignn/). Given atomistic structure in POSCAR format it predict formation energy, total energy per atom and bandgap using data trained on JARVIS-DFT dataset.

Similarly, a web-app for [ALIGNN-FF](https://jarvis.nist.gov/jalignnff/) for structure optimization is also available.

![JARVIS-ALIGNN](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/jalignn.PNG)



<a name="alignnff"></a>
ALIGNN-FF ASE Calculaor
-------------------------



[ASE calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html) provides interface to various codes. An example for ALIGNN-FF is give below. Note that there are multiple pretrained ALIGNN-FF models available, here we use the deafult_path model. As more accurate models are developed, they will be made available as well:

```
from alignn.ff.ff import AlignnAtomwiseCalculator,default_path
from jarvis.io.vasp.inputs import Poscar
import numpy as np
import matplotlib.pyplot as plt
model_path = default_path()
calc = AlignnAtomwiseCalculator(path=model_path)
# Source: https://www.ctcms.nist.gov/~knc6/static/JARVIS-DFT/JVASP-1002.xml
poscar="""Si2
1.0
3.3641499856336465 -2.5027128e-09 1.94229273881412
1.121382991333525 3.1717517190189715 1.9422927388141193
-2.5909987e-09 -1.8321133e-09 3.884586486670313
Si
2
Cartesian
3.92483875 2.77528125 6.7980237500000005
0.56069125 0.39646875 0.9711462500000001
"""
dx=np.arange(-0.1, 0.1, 0.01)
atoms=Poscar.from_string(poscar).atoms
print(atoms)
y = []
vol = []
for i in dx:
    struct = atoms.strain_atoms(i)
    struct_ase=struct.ase_converter()
    struct_ase.calc=calc
    y.append(struct_ase.get_potential_energy())
    vol.append(struct.volume)



plt.plot(vol,y,'-o')
plt.xlabel('Volume ($\AA^3$)')
plt.ylabel('Total energy (eV)')
plt.savefig('Si_JVASP-1002.png')
plt.close()
```






<a name="performances"></a>

Performances
-------------------------

Please refer to [JARVIS-Leaderboard](https://pages.nist.gov/jarvis_leaderboard/) to check the performance of ALIGNN models on several databases.

### 1) On JARVIS-DFT 2021 dataset (classification)

|     Model                               |     Threshold         |     ALIGNN    |
|-----------------------------------------|-----------------------|---------------|
|     Metal/non-metal classifier (OPT)    |     0.01 eV           |     0.92      |
|     Metal/non-metal classifier (MBJ)    |     0.01 eV           |     0.92      |
|     Magnetic/non-Magnetic classifier    |     0.05 µB           |     0.91      |
|     High/low SLME                       |     10 %              |     0.83      |
|     High/low spillage                   |     0.1               |     0.80      |
|     Stable/unstable (ehull)             |     0.1 eV            |     0.94      |
|     High/low-n-Seebeck                  |     -100 µVK<sup>-1</sup>       |     0.88      |
|     High/low-p-Seebeck                  |     100 µVK<sup>-1</sup>         |     0.92      |
|     High/low-n-powerfactor              |     1000 µW(mK<sup>2</sup>)<sup>-1</sup>    |     0.74      |
|     High/low-p-powerfactor              |     1000µW(mK<sup>2</sup>)<sup>-1</sup>     |     0.74      |


### 2) On JARVIS-DFT 2021 dataset (regression)

|     Property                                 |     Units                 |     MAD       |     CFID      |     CGCNN    |     ALIGNN    |     MAD: MAE    |
|----------------------------------------------|---------------------------|---------------|---------------|--------------|---------------|-----------------|
|     Formation   energy                       |     eV(atom)<sup>-1</sup>           |     0.86      |     0.14      |     0.063    |     0.033     |     26.06       |
|     Bandgap (OPT)                            |     eV                    |     0.99      |     0.30      |     0.20     |     0.14      |     7.07        |
|     Total   energy                           |     eV(atom)<sup>-1</sup>           |     1.78      |     0.24      |     0.078    |     0.037     |     48.11       |
|     Ehull                                    |     eV                    |     1.14      |     0.22      |     0.17     |     0.076     |     15.00       |
|     Bandgap   (MBJ)                          |     eV                    |     1.79      |     0.53      |     0.41     |     0.31      |     5.77        |
|     Kv                                       |     GPa                   |     52.80     |     14.12     |     14.47    |     10.40     |     5.08        |
|     Gv                                       |     GPa                   |     27.16     |     11.98     |     11.75    |     9.48      |     2.86        |
|     Mag. mom                                 |     µB                    |     1.27      |     0.45      |     0.37     |     0.26      |     4.88        |
|     SLME   (%)                               |     No   unit             |     10.93     |     6.22      |     5.66     |     4.52      |     2.42        |
|     Spillage                                 |     No unit               |     0.52      |     0.39      |     0.40     |     0.35      |     1.49        |
|     Kpoint-length                            |     Å                     |     17.88     |     9.68      |     10.60    |     9.51      |     1.88        |
|     Plane-wave cutoff                        |     eV                    |     260.4     |     139.4     |     151.0    |     133.8     |     1.95        |
|     єx   (OPT)                               |     No   unit             |     57.40     |     24.83     |     27.17    |     20.40     |     2.81        |
|     єy (OPT)                                 |     No unit               |     57.54     |     25.03     |     26.62    |     19.99     |     2.88        |
|     єz   (OPT)                               |     No   unit             |     56.03     |     24.77     |     25.69    |     19.57     |     2.86        |
|     єx (MBJ)                                 |     No unit               |     64.43     |     30.96     |     29.82    |     24.05     |     2.68        |
|     єy   (MBJ)                               |     No   unit             |     64.55     |     29.89     |     30.11    |     23.65     |     2.73        |
|     єz (MBJ)                                 |     No unit               |     60.88     |     29.18     |     30.53    |     23.73     |     2.57        |
|     є   (DFPT:elec+ionic)                    |     No   unit             |     45.81     |     43.71     |     38.78    |     28.15     |     1.63        |
|     Max. piezoelectric strain coeff (dij)    |     CN<sup>-1</sup>                  |     24.57     |     36.41     |     34.71    |     20.57     |     1.19        |
|     Max.   piezo. stress coeff (eij)         |     Cm<sup>-2</sup>                 |     0.26      |     0.23      |     0.19     |     0.147     |     1.77        |
|     Exfoliation energy                       |     meV(atom)<sup>-1</sup>          |     62.63     |     63.31     |     50.0     |     51.42     |     1.22        |
|     Max. EFG                                 |     10<sup>21</sup> Vm<sup>-2</sup>            |     43.90     |     24.54     |     24.7     |     19.12     |     2.30        |
|     avg. me                                  |     electron mass unit    |     0.22      |     0.14      |     0.12     |     0.085     |     2.59        |
|     avg. mh                                  |     electron mass unit    |     0.41      |     0.20      |     0.17     |     0.124     |     3.31        |
|     n-Seebeck                                |     µVK<sup>-1</sup>                 |     113.0     |     56.38     |     49.32    |     40.92     |     2.76        |
|     n-PF                                     |     µW(mK<sup>2</sup>)<sup>-1</sup>            |     697.80    |     521.54    |     552.6    |     442.30    |     1.58        |
|     p-Seebeck                                |     µVK<sup>-1</sup>                |     166.33    |     62.74     |     52.68    |     42.42     |     3.92        |
|     p-PF                                     |     µW(mK<sup>2</sup>)<sup>-1</sup>            |     691.67    |     505.45    |     560.8    |     440.26    |     1.57        |


### 3) On Materials project 2018 dataset

The results from models other than ALIGNN are reported as given in corresponding papers, not necessarily reproduced by us.

|     Prop    |     Unit      |     MAD     |     CFID     |     CGCNN    |     MEGNet    |     SchNet    |     ALIGNN    |     MAD:MAE    |
|-------------|---------------|-------------|--------------|--------------|---------------|---------------|---------------|----------------|
|     Ef      |     eV(atom)<sup>-1</sup>    |     0.93    |     0.104    |     0.039    |     0.028     |     0.035     |     0.022     |     42.27      |
|     Eg      |     eV        |     1.35    |     0.434    |     0.388    |     0.33      |     -         |     0.218     |     6.19       |



### 4) On QM9 dataset

Note the [issue](https://github.com/usnistgov/alignn/issues/54) related to QM9 dataset. The results from models other than ALIGNN are reported as given in corresponding papers, not necessarily reproduced by us. These models were trained with same parameters as solid-state databases but for 1000 epochs.

| Target | Units | SchNet | MEGNet  | DimeNet++ | ALIGNN |
|:------:|-------|--------|---------|-----------|--------|
|  HOMO  | eV    |  0.041 |  0.043  |   0.0246  | 0.0214 |
|  LUMO  | eV    |  0.034 |  0.044  |   0.0195  | 0.0195 |
|   Gap  | eV    |  0.063 |  0.066  |   0.0326  | 0.0381 |
|  ZPVE  | eV    | 0.0017 | 0.00143 |  0.00121  | 0.0031 |
|    µ   | Debye |  0.033 |   0.05  |   0.0297  | 0.0146 |
|    α   | Bohr<sup>3</sup> |  0.235 |  0.081  |   0.0435  | 0.0561 |
|    R<sup>2</sup>  | Bohr<sup>2</sup> |  0.073 |  0.302  |   0.331   | 0.5432 |
|   U0   | eV    |  0.014 |  0.012  |  0.00632  | 0.0153 |
|    U   | eV    |  0.019 |  0.013  |  0.00628  | 0.0144 |
|    H   | eV    |  0.014 |  0.012  |  0.00653  | 0.0147 |
|    G   | eV    |  0.014 |  0.012  |  0.00756  | 0.0144 |


### 5) On hMOF dataset

| Property           | Unit            | MAD     | MAE    | MAD:MAE | R<sup>2</sup> | RMSE   |
|--------------------|-----------------|---------|--------|---------|-------|--------|
| Grav. surface area | m<sup>2 </sup>g<sup>-1</sup>  | 1430.82 | 91.15  | 15.70   | 0.99  | 180.89 |
| Vol. surface area  | m<sup>2 </sup>cm<sup>-3</sup> | 561.44  | 107.81 | 5.21    | 0.91  | 229.24 |
| Void fraction      | No unit         | 0.16    | 0.017  | 9.41    | 0.98  | 0.03   |
| LCD                | Å   | 3.44    | 0.75   | 4.56    | 0.83  | 1.83   |
| PLD                | Å  | 3.55    | 0.92   | 3.86    | 0.78  | 2.12   |
| All adsp           | mol kg<sup>-1</sup>   | 1.70    | 0.18   | 9.44    | 0.95  | 0.49   |
| Adsp at 0.01bar    | mol kg<sup>-1</sup>  | 0.12    | 0.04   | 3.00    | 0.77  | 0.11   |
| Adsp at 2.5bar     | mol kg<sup>-1</sup>   | 2.16    | 0.48   | 4.50    | 0.90  | 0.97   |


### 6) On qMOF dataset

MAE on electronic bandgap 0.20 eV

### 7) On OMDB dataset

coming soon!

### 8) On HOPV dataset

coming soon!

### 9) On QETB dataset

coming soon!

### 10) On OpenCatalyst dataset

[On 10k dataset](https://github.com/Open-Catalyst-Project/ocp/blob/main/MODELS.md#is2re-models):

|     DataSplit                                |     CGCNN                 |  DimeNet      |     SchNet    | DimeNet++    |     ALIGNN    |     MAD: MAE    |
|----------------------------------------------|---------------------------|---------------|---------------|--------------|---------------|-----------------|
|     10k                                      |     0.988                 |   1.0117      |    1.059      |    0.8837    |     0.61      |     -           |




<a name="notes"></a>
Useful notes (based on some of the queries we received)
---------------------------------------------------------

1) If you are using GPUs, make sure you have a compatible dgl-cuda version installed, for example: dgl-cu101 or dgl-cu111, so e.g. `pip install dgl-cu111` .
2) While comnventional '.cif' and '.pdb' files can be read using jarvis-tools, for complex files you might have to install `cif2cell` and `pytraj` respectively i.e.`pip install cif2cell==2.0.0a3` and `conda install -c ambermd pytraj`.
3) Make sure you use `batch_size` as 32 or 64 for large datasets, and not 2 as given in the example config file, else it will take much longer to train, and performnce might drop a lot.
4) Note that `train_alignn.py` and `pretrained.py` in alignn folder are actually python executable scripts. So, even if you don't provide absolute path of these scripts, they should work.
5) Learn about the issue with QM9 results here: https://github.com/usnistgov/alignn/issues/54
6) Make sure you have `pandas` version as >1.2.3.
7) Starting March 2024, pytroch-ignite dependency will be removed to enable conda-forge build.


<a name="refs"></a>
References
-----------------

1) [Atomistic Line Graph Neural Network for improved materials property predictions](https://www.nature.com/articles/s41524-021-00650-1)
2) [Prediction of the Electron Density of States for Crystalline Compounds with Atomistic Line Graph Neural Networks (ALIGNN)](https://link.springer.com/article/10.1007/s11837-022-05199-y)
3) [Recent advances and applications of deep learning methods in materials science](https://www.nature.com/articles/s41524-022-00734-6)
4) [Designing High-Tc Superconductors with BCS-inspired Screening, Density Functional Theory and Deep-learning](https://arxiv.org/abs/2205.00060)
5) [A Deep-learning Model for Fast Prediction of Vacancy Formation in Diverse Materials](https://arxiv.org/abs/2205.08366)
6) [Graph neural network predictions of metal organic framework CO2 adsorption properties](https://www.sciencedirect.com/science/article/pii/S092702562200163X)
7) [Rapid Prediction of Phonon Structure and Properties using an Atomistic Line Graph Neural Network (ALIGNN)](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.7.023803)
8) [Unified graph neural network force-field for the periodic table](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00096b)
9) [Large Scale Benchmark of Materials Design Methods](https://www.nature.com/articles/s41524-024-01259-w)
10) [Prediction of Magnetic Properties in van der Waals Magnets using Graph Neural Networks](https://doi.org/10.1103/PhysRevMaterials.8.114002)
11) [CHIPS-FF: Benchmarking universal force-fields](https://github.com/usnistgov/chipsff)


Please see detailed publications list [here](https://jarvis-tools.readthedocs.io/en/master/publications.html).

<a name="contrib"></a>
How to contribute
-----------------

For detailed instructions, please see [Contribution instructions](https://github.com/usnistgov/jarvis/blob/master/Contribution.rst)

<a name="corres"></a>
Correspondence
--------------------

Please report bugs as Github issues (https://github.com/usnistgov/alignn/issues) or email to kamal.choudhary@nist.gov.

<a name="fund"></a>
Funding support
--------------------

NIST-MGI (https://www.nist.gov/mgi)

NIST-CHIPS (https://www.nist.gov/chips)

Code of conduct
--------------------

Please see [Code of conduct](https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md)
