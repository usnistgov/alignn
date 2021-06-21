
![alt text](https://github.com/usnistgov/alignn/actions/workflows/main.yml/badge.svg)
# ALIGNN
Atomistic Line Graph Neural Network (https://arxiv.org/abs/2106.01829)

Installation
-------------------------
First create a conda environment:
Install miniconda environment from https://conda.io/miniconda.html
```
bash Miniconda3-latest-Linux-x86_64.sh (for linux)
bash Miniconda3-latest-MacOSX-x86_64.sh (for Mac)
Download 32/64 bit python 3.6 miniconda exe and install (for windows)
Now, let's make a conda environment, say "version", choose other name as you like::
conda create --name version python=3.8
source activate version
```
Now, let's install the package
```
git clone https://github.com/usnistgov/alignn.git
cd alignn
python setup.py develop
```
Example
---------
```
python alignn/scripts/train_folder.py --root_dir "alignn/examples/sample_data" --config "alignn/examples/sample_data/config_example_regrssion.json"
```

You can also try multiple example scripts to run multiple dataset training. Look into the 'scripts' folder.
