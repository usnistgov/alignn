
![alt text](https://github.com/usnistgov/alignn/actions/workflows/main.yml/badge.svg)
# ALIGNN
Atomistic Line Graph Neural Network (https://arxiv.org/abs/2106.01829)

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
Download 32/64 bit python 3.6 miniconda exe and install (for windows)
Now, let's make a conda environment, say "version", choose other name as you like::
```
conda create --name version python=3.8
source activate version
```

Now, let's install the package:
```
git clone https://github.com/usnistgov/alignn.git
cd alignn
python setup.py develop
```
Examples
---------
Users can keep their structure files in POSCAR, .cif, or .xyz files in a directory. In the examples below we will use POSCAR format files. In the same directory, there should be id_prop.csv file.
In this id_prop.csv, the filenames, and correponding target values are kept in comma separated values (csv) format.
Here is an example of training OptB88vdw bandgaps of 50 materials from JARVIS-DFT. The example is created using the script provided in the script folder.
Users can modify the script more than 50 data, or make their own dataset in this format. The dataset in split in 80:10:10 as training-validation-test set. To change the split proportion and other parameters, change the config_example.json file. Sometimes, we want to train on certain sets and val/test on another dataset. For such cases, set n_train, n_val, n_test manually in the config_example.json and also set keep_data_order as True there.  With the configuration parameters given in config_example.json, the model is trained.

```
python alignn/scripts/train_folder.py --root_dir "alignn/examples/sample_data" --config "alignn/examples/sample_data/config_example.json"
```
While the above example is for regression, the follwoing example shows a classification task for metal/non-metal based on the above bandgap values. We transform the dataset
into 1 or 0 based on a threshold of 0.01 eV (controlled by the parameter, 'classification_threshold') and train a similar classification model.
```
python alignn/scripts/train_folder.py --root_dir "alignn/examples/sample_data" --config "alignn/examples/sample_data/config_example.json"
```
While the above example regression was for single-output values, we can train multi-output regression models as well.
An example is given below for training formation energy per atom, bandgap and total energy per atom simulataneously. The script to generate the example data is provided in the script folder of the sample_data_multi_prop. 
Another example of training electron and phonon density of states is provided also.
```
python alignn/scripts/train_folder.py --root_dir "alignn/examples/sample_data_multi_prop" --config "alignn/examples/sample_data/config_example.json"
```

Users can also try multiple example scripts to run multiple dataset training. Look into the 'alignn/scripts' folder. 
These scripts automatically download datasets from jarvis.db.fighshare module in JARVIS-Tools and train several models. Make sure you specify your specific queuing system details in the scripts. 
