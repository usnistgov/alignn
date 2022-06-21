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

# Table of Contents

* [Introduction](#intro)
* [Installation](#install)
* [Examples](#example)
* [Pre-trained models](#pretrained)
* [Quick start using colab](#colab)
* [JARVIS-ALIGNN webapp](#webapp)
* [Peformances on a few datasets](#performances)
* [Useful notes](#notes)
* [References](#refs)
* [How to contribute](#contrib)
* [Correspondence](#corres)
* [Funding support](#fund)

<a name="intro"></a>

# ALIGNN (Introduction)

The Atomistic Line Graph Neural Network (<https://www.nature.com/articles/s41524-021-00650-1>)  introduces a new graph convolution layer that explicitly models both two and three body interactions in atomistic systems.

This is achieved by composing two edge-gated graph convolution layers, the first applied to the atomistic line graph *L(g)* (representing triplet interactions) and the second applied to the atomistic bond graph *g* (representing pair interactions).

The atomistic graph *g* consists of a node for each atom *i* (with atom/node representations *h<sub>i</sub>*), and one edge for each atom pair within a cutoff radius (with bond/pair representations *e<sub>ij</sub>*).

The atomistic line graph *L(g)* represents relationships between atom triplets: it has nodes corresponding to bonds (sharing representations *e<sub>ij</sub>* with those in *g*) and edges corresponding to bond angles (with angle/triplet representations *t<sub>ijk</sub>*).

The line graph convolution updates the triplet representations and the pair representations; the direct graph convolution further updates the pair representations and the atom representations.

![ALIGNN layer schematic](https://github.com/usnistgov/alignn/blob/main/alignn/tex/alignn2.png)

<a name="install"></a>
Installation
-------------------------

First create a conda environment:
Install miniconda environment from <https://conda.io/miniconda.html>
Based on your system requirements, you'll get a file something like 'Miniconda3-latest-XYZ'.

Now,

```sh
bash Miniconda3-latest-Linux-x86_64.sh (for linux)
bash Miniconda3-latest-MacOSX-x86_64.sh (for Mac)
```

Download 32/64 bit python 3.8 miniconda exe and install (for windows)
Now, let's make a conda environment, say "version", choose other name as you like::

```sh
conda create --name version python=3.8
source activate version
```

### Method 1 (using setup.py)

Now, let's install the package:

```sh
git clone https://github.com/usnistgov/alignn.git
cd alignn
python setup.py develop
```

For using GPUs/CUDA, install dgl-cu101 or dgl-cu111 based on the CUDA version available on your system, e.g.

```sh
pip install dgl-cu111
```

### Method 2 (using pypi)

As an alternate method, ALIGNN can also be installed using `pip` command as follows:

```sh
pip install alignn dgl-cu111
```

<a name="example"></a>
Examples
---------

### Dataset

The main script to train model is `train_folder.py`. A user needs at least the following info to train a model: 1) `id_prop.csv` with name of the file and corresponding value, 2) `config_example.json` a config file with training and hyperparameters.

Users can keep their structure files in `POSCAR`, `.cif`, `.xyz` or `.pdb` files in a directory. In the examples below we will use POSCAR format files. In the same directory, there should be an `id_prop.csv` file.

In this directory, `id_prop.csv`, the filenames, and corresponding target values are kept in `comma separated values (csv) format`.

Here is an example of training OptB88vdw bandgaps of 50 materials from JARVIS-DFT database. The example is created using the [generate_sample_data_reg.py](https://github.com/usnistgov/alignn/blob/main/alignn/examples/sample_data/scripts/generate_sample_data_reg.py) script. Users can modify the script for more than 50 data, or make their own dataset in this format. For list of available datasets see [Databases](https://jarvis-tools.readthedocs.io/en/master/databases.html).

The dataset in split in 80:10:10 as training-validation-test set (controlled by `train_ratio, val_ratio, test_ratio`) . To change the split proportion and other parameters, change the [config_example.json](https://github.com/usnistgov/alignn/blob/main/alignn/examples/sample_data/config_example.json) file. If, users want to train on certain sets and val/test on another dataset, set `n_train`, `n_val`, `n_test` manually in the `config_example.json` and also set `keep_data_order` as True there so that random shuffle is disabled.

A brief help guide (`-h`) can be obtained as follows.

```sh
train_folder.py -h
```

### Regression example

Now, the model is trained as follows. Please increase the `batch_size` parameter to something like 32 or 64 in `config_example.json` for general trainings.

```sh
train_folder.py --root_dir "alignn/examples/sample_data" --config "alignn/examples/sample_data/config_example.json" --output_dir=temp
```

### Classification example

While the above example is for regression, the following example shows a classification task for metal/non-metal based on the above bandgap values. We transform the dataset
into 1 or 0 based on a threshold of 0.01 eV (controlled by the parameter, `classification_threshold`) and train a similar classification model. Currently, the script allows binary classification tasks only.

```sh
train_folder.py --root_dir "alignn/examples/sample_data" --classification_threshold 0.01 --config "alignn/examples/sample_data/config_example.json" --output_dir=temp
```

### Multi-output model example

While the above example regression was for single-output values, we can train multi-output regression models as well.
An example is given below for training formation energy per atom, bandgap and total energy per atom simultaneously. The script to generate the example data is provided in the script folder of the sample_data_multi_prop. Another example of training electron and phonon density of states is provided also.

```sh
train_folder.py --root_dir "alignn/examples/sample_data_multi_prop" --config "alignn/examples/sample_data/config_example.json" --output_dir=temp
```

### Automated model training

Users can try training using multiple example scripts to run multiple dataset (such as JARVIS-DFT, Materials project, QM9_JCTC etc.). Look into the [alignn/scripts/train_*.py](https://github.com/usnistgov/alignn/tree/main/alignn/scripts) folder. This is done primarily to make the trainings more automated rather than making folder/ csv files etc.
These scripts automatically download datasets from [Databases in jarvis-tools](https://jarvis-tools.readthedocs.io/en/master/databases.html) and train several models. Make sure you specify your specific queuing system details in the scripts.

<a name="pretrained"></a>
Using pre-trained models
-------------------------

All the trained models are distributed on [figshare](https://figshare.com/projects/ALIGNN_models/126478) and this [pretrained.py script](https://github.com/usnistgov/alignn/blob/develop/alignn/pretrained.py) can be applied to use them. These models can be used to directly make predictions.

A brief help section (`-h`) is shown using:

```sh
pretrained.py -h
```

An example of prediction formation energy per atom using JARVIS-DFT dataset trained model is shown below:

```sh
pretrained.py --model_name jv_formation_energy_peratom_alignn --file_format poscar --file_path alignn/examples/sample_data/POSCAR-JVASP-10.vasp
```

<a name="colab"></a>
Quick start using GoogleColab notebook example
-----------------------------------------------

The following [notebook](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Training_ALIGNN_model_example.ipynb) provides an example of 1) installing ALIGNN model, 2) training the example data and 3) using the pretrained models. For this example, you don't need to install alignn package on your local computer/cluster, it requires a gmail account to login. Learn more about Google colab [here](https://colab.research.google.com/notebooks/intro.ipynb).

[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Training_ALIGNN_model_example.ipynb)

<a name="webapp"></a>
Web-app
------------

A basic web-app is for direct-prediction available at [JARVIS-ALIGNN app](https://jarvis.nist.gov/jalignn/). Given atomistic structure in POSCAR format it predict formation energy, total energy per atom and bandgap using data trained on JARVIS-DFT dataset.

![JARVIS-ALIGNN](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/jalignn.PNG)

<a name="performances"></a>

Performances
-------------------------

### 1) On JARVIS-DFT 2021 dataset (classification)

| Model                            | Threshold                            | ALIGNN |
| -------------------------------- | ------------------------------------ | ------ |
| Metal/non-metal classifier (OPT) | 0.01 eV                              | 0.92   |
| Metal/non-metal classifier (MBJ) | 0.01 eV                              | 0.92   |
| Magnetic/non-Magnetic classifier | 0.05 µB                              | 0.91   |
| High/low SLME                    | 10 %                                 | 0.83   |
| High/low spillage                | 0.1                                  | 0.80   |
| Stable/unstable (ehull)          | 0.1 eV                               | 0.94   |
| High/low-n-Seebeck               | -100 µVK<sup>-1</sup>                | 0.88   |
| High/low-p-Seebeck               | 100 µVK<sup>-1</sup>                 | 0.92   |
| High/low-n-powerfactor           | 1000 µW(mK<sup>2</sup>)<sup>-1</sup> | 0.74   |
| High/low-p-powerfactor           | 1000µW(mK<sup>2</sup>)<sup>-1</sup>  | 0.74   |

### 2) On JARVIS-DFT 2021 dataset (regression)

| Property                              | Units                           | MAD    | CFID   | CGCNN | ALIGNN | MAD: MAE |
| ------------------------------------- | ------------------------------- | ------ | ------ | ----- | ------ | -------- |
| Formation   energy                    | eV(atom)<sup>-1</sup>           | 0.86   | 0.14   | 0.063 | 0.033  | 26.06    |
| Bandgap (OPT)                         | eV                              | 0.99   | 0.30   | 0.20  | 0.14   | 7.07     |
| Total   energy                        | eV(atom)<sup>-1</sup>           | 1.78   | 0.24   | 0.078 | 0.037  | 48.11    |
| Ehull                                 | eV                              | 1.14   | 0.22   | 0.17  | 0.076  | 15.00    |
| Bandgap   (MBJ)                       | eV                              | 1.79   | 0.53   | 0.41  | 0.31   | 5.77     |
| Kv                                    | GPa                             | 52.80  | 14.12  | 14.47 | 10.40  | 5.08     |
| Gv                                    | GPa                             | 27.16  | 11.98  | 11.75 | 9.48   | 2.86     |
| Mag. mom                              | µB                              | 1.27   | 0.45   | 0.37  | 0.26   | 4.88     |
| SLME   (%)                            | No   unit                       | 10.93  | 6.22   | 5.66  | 4.52   | 2.42     |
| Spillage                              | No unit                         | 0.52   | 0.39   | 0.40  | 0.35   | 1.49     |
| Kpoint-length                         | Å                               | 17.88  | 9.68   | 10.60 | 9.51   | 1.88     |
| Plane-wave cutoff                     | eV                              | 260.4  | 139.4  | 151.0 | 133.8  | 1.95     |
| єx   (OPT)                            | No   unit                       | 57.40  | 24.83  | 27.17 | 20.40  | 2.81     |
| єy (OPT)                              | No unit                         | 57.54  | 25.03  | 26.62 | 19.99  | 2.88     |
| єz   (OPT)                            | No   unit                       | 56.03  | 24.77  | 25.69 | 19.57  | 2.86     |
| єx (MBJ)                              | No unit                         | 64.43  | 30.96  | 29.82 | 24.05  | 2.68     |
| єy   (MBJ)                            | No   unit                       | 64.55  | 29.89  | 30.11 | 23.65  | 2.73     |
| єz (MBJ)                              | No unit                         | 60.88  | 29.18  | 30.53 | 23.73  | 2.57     |
| є   (DFPT:elec+ionic)                 | No   unit                       | 45.81  | 43.71  | 38.78 | 28.15  | 1.63     |
| Max. piezoelectric strain coeff (dij) | CN<sup>-1</sup>                 | 24.57  | 36.41  | 34.71 | 20.57  | 1.19     |
| Max.   piezo. stress coeff (eij)      | Cm<sup>-2</sup>                 | 0.26   | 0.23   | 0.19  | 0.147  | 1.77     |
| Exfoliation energy                    | meV(atom)<sup>-1</sup>          | 62.63  | 63.31  | 50.0  | 51.42  | 1.22     |
| Max. EFG                              | 10<sup>21</sup> Vm<sup>-2</sup> | 43.90  | 24.54  | 24.7  | 19.12  | 2.30     |
| avg. me                               | electron mass unit              | 0.22   | 0.14   | 0.12  | 0.085  | 2.59     |
| avg. mh                               | electron mass unit              | 0.41   | 0.20   | 0.17  | 0.124  | 3.31     |
| n-Seebeck                             | µVK<sup>-1</sup>                | 113.0  | 56.38  | 49.32 | 40.92  | 2.76     |
| n-PF                                  | µW(mK<sup>2</sup>)<sup>-1</sup> | 697.80 | 521.54 | 552.6 | 442.30 | 1.58     |
| p-Seebeck                             | µVK<sup>-1</sup>                | 166.33 | 62.74  | 52.68 | 42.42  | 3.92     |
| p-PF                                  | µW(mK<sup>2</sup>)<sup>-1</sup> | 691.67 | 505.45 | 560.8 | 440.26 | 1.57     |

### 3) On Materials project 2018 dataset

The results from models other than ALIGNN are reported as given in corresponding papers, not necessarily reproduced by us. You can also refer to [MatBench](https://matbench.materialsproject.org/) project to check the performance of ALIGNN model on the Materials project and other databases.

| Prop | Unit                  | MAD  | CFID  | CGCNN | MEGNet | SchNet | ALIGNN | MAD:MAE |
| ---- | --------------------- | ---- | ----- | ----- | ------ | ------ | ------ | ------- |
| Ef   | eV(atom)<sup>-1</sup> | 0.93 | 0.104 | 0.039 | 0.028  | 0.035  | 0.022  | 42.27   |
| Eg   | eV                    | 1.35 | 0.434 | 0.388 | 0.33   | -      | 0.218  | 6.19    |

### 4) On QM9 dataset

Note the [issue](https://github.com/usnistgov/alignn/issues/54) related to QM9 dataset. The results from models other than ALIGNN are reported as given in corresponding papers, not necessarily reproduced by us. These models were trained with same parameters as solid-state databases but for 1000 epochs.

|    Target     | Units            | SchNet | MEGNet  | DimeNet++ | ALIGNN |
| :-----------: | ---------------- | ------ | ------- | --------- | ------ |
|     HOMO      | eV               | 0.041  | 0.043   | 0.0246    | 0.0214 |
|     LUMO      | eV               | 0.034  | 0.044   | 0.0195    | 0.0195 |
|      Gap      | eV               | 0.063  | 0.066   | 0.0326    | 0.0381 |
|     ZPVE      | eV               | 0.0017 | 0.00143 | 0.00121   | 0.0031 |
|       µ       | Debye            | 0.033  | 0.05    | 0.0297    | 0.0146 |
|       α       | Bohr<sup>3</sup> | 0.235  | 0.081   | 0.0435    | 0.0561 |
| R<sup>2</sup> | Bohr<sup>2</sup> | 0.073  | 0.302   | 0.331     | 0.5432 |
|      U0       | eV               | 0.014  | 0.012   | 0.00632   | 0.0153 |
|       U       | eV               | 0.019  | 0.013   | 0.00628   | 0.0144 |
|       H       | eV               | 0.014  | 0.012   | 0.00653   | 0.0147 |
|       G       | eV               | 0.014  | 0.012   | 0.00756   | 0.0144 |

### 5) On hMOF dataset

| Property           | Unit                          | MAD     | MAE    | MAD:MAE | R<sup>2</sup> | RMSE   |
| ------------------ | ----------------------------- | ------- | ------ | ------- | ------------- | ------ |
| Grav. surface area | m<sup>2 </sup>g<sup>-1</sup>  | 1430.82 | 91.15  | 15.70   | 0.99          | 180.89 |
| Vol. surface area  | m<sup>2 </sup>cm<sup>-3</sup> | 561.44  | 107.81 | 5.21    | 0.91          | 229.24 |
| Void fraction      | No unit                       | 0.16    | 0.017  | 9.41    | 0.98          | 0.03   |
| LCD                | Å                             | 3.44    | 0.75   | 4.56    | 0.83          | 1.83   |
| PLD                | Å                             | 3.55    | 0.92   | 3.86    | 0.78          | 2.12   |
| All adsp           | mol kg<sup>-1</sup>           | 1.70    | 0.18   | 9.44    | 0.95          | 0.49   |
| Adsp at 0.01bar    | mol kg<sup>-1</sup>           | 0.12    | 0.04   | 3.00    | 0.77          | 0.11   |
| Adsp at 2.5bar     | mol kg<sup>-1</sup>           | 2.16    | 0.48   | 4.50    | 0.90          | 0.97   |

### 6) On qMOF dataset

MAE on electronic bandgap 0.20 eV

### 7) On OMDB dataset

coming soon!

### 8) On HOPV dataset

coming soon!

### 9) On QETB dataset

coming soon!

### 10) On OpenCatalyst dataset

coming soon!

<a name="notes"></a>
Useful notes (based on some of the queries we received)
---------------------------------------------------------

1) If you are using GPUs, make sure you have a compatible dgl-cuda version installed, for example: dgl-cu101 or dgl-cu111, so e.g. `pip install dgl-cu111` .
2) The undirected graph and its line graph is constructured in `jarvis-tools` package using [jarvis.core.graphs](https://github.com/usnistgov/jarvis/blob/master/jarvis/core/graphs.py#L197)
3) While comnventional '.cif' and '.pdb' files can be read using jarvis-tools, for complex files you might have to install `cif2cell` and `pytraj` respectively i.e.`pip install cif2cell==2.0.0a3` and `conda install -c ambermd pytraj`.
4) Make sure you use `batch_size` as 32 or 64 for large datasets, and not 2 as given in the example config file, else it will take much longer to train, and performnce might drop a lot.
5) Note that `train_folder.py` and `pretrained.py` in alignn folder are actually python executable scripts. So, even if you don't provide absolute path of these scripts, they should work.
6) Learn about the issue with QM9 results here: <https://github.com/usnistgov/alignn/issues/54>
7) Make sure you have `pandas` version as 1.2.3.

<a name="refs"></a>
References
-----------------

Please see detailed publications list [here](https://jarvis-tools.readthedocs.io/en/master/publications.html).

<a name="contrib"></a>
How to contribute
-----------------

For detailed instructions, please see [Contribution instructions](https://github.com/usnistgov/jarvis/blob/master/Contribution.rst)

<a name="corres"></a>
Correspondence
--------------------

Please report bugs as Github issues (<https://github.com/usnistgov/alignn/issues>) or email to kamal.choudhary@nist.gov.

<a name="fund"></a>
Funding support
--------------------

NIST-MGI (<https://www.nist.gov/mgi>).

Code of conduct
--------------------

Please see [Code of conduct](https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md)
