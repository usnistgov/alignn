[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Training_ALIGNN_model_example.ipynb)
![alt text](https://github.com/usnistgov/alignn/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/usnistgov/alignn/branch/main/graph/badge.svg?token=S5X4OYC80V)](https://codecov.io/gh/usnistgov/alignn)
[![PyPI version](https://badge.fury.io/py/alignn.svg)](https://badge.fury.io/py/alignn)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/usnistgov/alignn)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/usnistgov/alignn)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/usnistgov/alignn)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/formation-energy-on-materials-project)](https://paperswithcode.com/sota/formation-energy-on-materials-project?p=atomistic-line-graph-neural-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/band-gap-on-materials-project)](https://paperswithcode.com/sota/band-gap-on-materials-project?p=atomistic-line-graph-neural-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/formation-energy-on-qm9)](https://paperswithcode.com/sota/formation-energy-on-qm9?p=atomistic-line-graph-neural-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/formation-energy-on-jarvis-dft-formation)](https://paperswithcode.com/sota/formation-energy-on-jarvis-dft-formation?p=atomistic-line-graph-neural-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/atomistic-line-graph-neural-network-for/band-gap-on-jarvis-dft)](https://paperswithcode.com/sota/band-gap-on-jarvis-dft?p=atomistic-line-graph-neural-network-for)
[![Downloads](https://pepy.tech/badge/alignn)](https://pepy.tech/project/alignn)


# ALIGNN
The Atomistic Line Graph Neural Network (https://www.nature.com/articles/s41524-021-00650-1)  introduces a new graph convolution layer that explicitly models both two and three body interactions in atomistic systems.

This is achieved by composing two edge-gated graph convolution layers, the first applied to the atomistic line graph *L(g)* (representing triplet interactions) and the second applied to the atomistic bond graph *g* (representing pair interactions).

The atomistic graph *g* consists of a node for each atom *i* (with atom/node representations *h<sub>i</sub>*), and one edge for each atom pair within a cutoff radius (with bond/pair representations *e<sub>ij</sub>*).

The atomistic line graph *L(g)* represents relationships between atom triplets: it has nodes corresponding to bonds (sharing representations *e<sub>ij</sub>* with those in *g*) and edges corresponding to bond angles (with angle/triplet representations *t<sub>ijk</sub>*).

The line graph convolution updates the triplet representations and the pair representations; the direct graph convolution further updates the pair representations and the atom representations.

![ALIGNN layer schematic](https://github.com/usnistgov/alignn/blob/main/alignn/tex/alignn2.png)


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
Download 32/64 bit python 3.8 miniconda exe and install (for windows)
Now, let's make a conda environment, say "version", choose other name as you like::
```
conda create --name version python=3.8
source activate version
```
#### Method 1 (using setup.py):

Now, let's install the package:
```
git clone https://github.com/usnistgov/alignn.git
cd alignn
python setup.py develop
```
For using GPUs/CUDA, install dgl-cu101 or dgl-cu111 based on the CUDA version available on your system, e.g.

```
pip install dgl-cu111
```

#### Method 2 (using pypi):

As an alternate method, ALIGNN can also be installed using `pip` command as follows:
```
pip install alignn dgl-cu111
```

Examples
---------

#### Dataset
Users can keep their structure files in `POSCAR`, `.cif`, `.xyz` or `.pdb` files in a directory. In the examples below we will use POSCAR format files. In the same directory, there should be an `id_prop.csv` file. 

In this directory, `id_prop.csv`, the filenames, and correponding target values are kept in `comma separated values (csv) format`.

Here is an example of training OptB88vdw bandgaps of 50 materials from JARVIS-DFT database. The example is created using the [generate_sample_data_reg.py](https://github.com/usnistgov/alignn/blob/main/alignn/examples/sample_data/scripts/generate_sample_data_reg.py) script. Users can modify the script for more than 50 data, or make their own dataset in this format. For list of available datasets see [Databases](https://jarvis-tools.readthedocs.io/en/master/databases.html).

The dataset in split in 80:10:10 as training-validation-test set (controlled by `train_ratio, val_ratio, test_ratio`) . To change the split proportion and other parameters, change the `config_example.json` file. If, users want to train on certain sets and val/test on another dataset, set `n_train`, `n_val`, `n_test` manually in the `config_example.json` and also set `keep_data_order` as True there so that random shuffle is disabled.

A brief help guide (`-h`) can be obtained as follows. 

```
!train_folder.py -h 
```
#### Regression example
Now, the model is trained as follows. Please increase the `batch_size` parameter to something like 32 or 64 in `config_example.json` for general trainings.

```
!train_folder.py --root_dir "alignn/examples/sample_data" --config "alignn/examples/sample_data/config_example.json" --output_dir=temp
```
#### Classification example
While the above example is for regression, the follwoing example shows a classification task for metal/non-metal based on the above bandgap values. We transform the dataset
into 1 or 0 based on a threshold of 0.01 eV (controlled by the parameter, `classification_threshold`) and train a similar classification model. Currently, the script allows binary classification tasks only.
```
!train_folder.py --root_dir "alignn/examples/sample_data" --classification_threshold 0.01 --config "alignn/examples/sample_data/config_example.json" --output_dir=temp
```

#### Multi-output model example
While the above example regression was for single-output values, we can train multi-output regression models as well.
An example is given below for training formation energy per atom, bandgap and total energy per atom simulataneously. The script to generate the example data is provided in the script folder of the sample_data_multi_prop. Another example of training electron and phonon density of states is provided also.
```
!train_folder.py --root_dir "alignn/examples/sample_data_multi_prop" --config "alignn/examples/sample_data/config_example.json" --output_dir=temp
```
#### Automated model training
Users can try training using multiple example scripts to run multiple dataset (such as JARVIS-DFT, Materials project, QM9_JCTC etc.). Look into the [alignn/scripts/train_*.py](https://github.com/usnistgov/alignn/tree/main/alignn/scripts) folder. This is done primarily to make the trainings more automated rather than making folder/ csv files etc.
These scripts automatically download datasets from [Databases in jarvis-tools](https://jarvis-tools.readthedocs.io/en/master/databases.html) and train several models. Make sure you specify your specific queuing system details in the scripts.

Using pretrained models
-------------------------

All the trained models are distributed on [figshare](https://figshare.com/projects/ALIGNN_models/126478) and this [pretrained.py script](https://github.com/usnistgov/alignn/blob/develop/alignn/pretrained.py) can be applied to use them. These models can be used to directly make predictions.

A brief help section is shown using:

```
!pretrained.py -h
```
An example of prediction formation energy per atom using JARVIS-DFT dataset trained model is shown below:

```
!pretrained.py --model_name jv_formation_energy_peratom_alignn --file_format poscar --file_path alignn/examples/sample_data/POSCAR-JVASP-10.vasp
```

GoogleColab notebook example
-----------------------------

The following notebook provides an example of 1) installing ALIGNN model, 2) training the example data and 3) using the pretrained models. For this example, you don't need to install alignn package on your local computer/cluster, it requires a gmail account to login. Learn more about Google colab [here](https://colab.research.google.com/notebooks/intro.ipynb).

[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Training_ALIGNN_model_example.ipynb)

Web-app
------------

A basic web-app is for direct-prediction available at [JARVIS-ALIGNN app](https://jarvis.nist.gov/jalignn/). Given atomistic structure in POSCAR format it predict formation energy, total energy per atom and bandgap using data trained on JARVIS-DFT dataset.

![JARVIS-ALIGNN](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/jalignn.PNG)

Performances
-------------------------

On QM9 dataset

![QM9](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/qm9.PNG)

On Materials project dataset

![MP](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/MP.PNG)

On JARVIS-DFT dataset (classification)

![JV-class](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/jvclass.PNG)

On JARVIS-DFT dataset (regression)

![JV-reg1](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/jv.PNG)
![JV-reg2](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/jv2.PNG)

On hMOF dataset

On qMOF dataset

On OMDB dataset

Om HOPV dataset

Notes:
1) If you are using GPUs, make sure you have a compatible dgl-cuda version installed, for example: dgl-cu101 or dgl-cu111.
2) The undirected graph and its line graph is constructured in `jarvis-tools` package using [jarvis.core.graphs](https://github.com/usnistgov/jarvis/blob/master/jarvis/core/graphs.py#L197)
3) While comnventional '.cif' and '.pdb' files can be read using jarvis-tools, for complex files you might have to install `cif2cell` and `pytraj` respectively i.e.`pip install cif2cell==2.0.0a3` and `conda install -c ambermd pytraj`.
4) Make sure you use `batch_size` as 32 or 64 for large datasets, and not 2 as given in the example config file, else it will take much longer to train, and performnce might drop a lot.
5) Note that `train_folder.py` and `pretrained.py` in alignn folder are actually python executable scripts. So, even if you don't provide absolute path of these scripts, they should work.

References
-----------------

Please see `Publications related to JARVIS infrastructure can be found here: <https://jarvis-tools.readthedocs.io/en/master/publications.html>`__

How to contribute
-----------------

.. image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square
        :target: http://makeapullrequest.com

For detailed instructions, please see `Contribution instructions <https://github.com/usnistgov/jarvis/blob/master/Contribution.rst>`__

Correspondence
--------------------

Please report bugs as Github issues (https://github.com/usnistgov/alignn/issues) or email to kamal.choudhary@nist.gov.

Funding support
--------------------

NIST-MGI (https://www.nist.gov/mgi).

Code of conduct
--------------------

Please see `Code of conduct <https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md>`__

