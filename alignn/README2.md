![Build Status](https://github.com/usnistgov/alignn/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/usnistgov/alignn/branch/main/graph/badge.svg?token=S5X4OYC80V)](https://codecov.io/gh/usnistgov/alignn)
[![PyPI version](https://badge.fury.io/py/alignn.svg)](https://badge.fury.io/py/alignn)
![Latest GitHub Tag](https://img.shields.io/github/v/tag/usnistgov/alignn)
![Code Size](https://img.shields.io/github/languages/code-size/usnistgov/alignn)
![Commit Activity](https://img.shields.io/github/commit-activity/y/usnistgov/alignn)
[![Downloads](https://pepy.tech/badge/alignn)](https://pepy.tech/project/alignn)

# ALIGNN & ALIGNN-FF

## Overview

The Atomistic Line Graph Neural Network (ALIGNN) introduces a novel graph convolution layer that captures both two-body and three-body interactions in atomistic systems, enhancing materials property prediction. ALIGNN uses a composition of two edge-gated graph convolutions: one on the atomistic line graph (triplet interactions), and one on the bond graph (pairwise interactions).

ALIGNN-FF is an extension tailored to force field modeling, trained on the JARVIS-DFT dataset (~75,000 materials and 4 million entries), supporting 89 elements for structural and chemical diversity. Users can fine-tune existing models or train from scratch on custom datasets.

![ALIGNN Layer Diagram](https://github.com/usnistgov/alignn/blob/develop/alignn/tex/schematic_lg.jpg)

## Table of Contents
- [Installation](#installation)
- [Examples](#examples)
- [Pre-trained Models](#pretrained-models)
- [Web Applications](#web-applications)
- [ALIGNN-FF & ASE Calculator](#alignn-ff--ase-calculator)
- [Performance Benchmarks](#performance-benchmarks)
- [Notes & Tips](#notes--tips)
- [References](#references)
- [Contribution Guide](#contribution-guide)
- [Contact](#contact)
- [Funding](#funding)

---

## Installation

You can install ALIGNN using GitHub (development), PyPI with `uv`, or via Conda/Miniforge.

### Step 1: Install Miniforge (Recommended Python Environment)
Download Miniforge installer from: https://github.com/conda-forge/miniforge
Choose the installer matching your OS (e.g., `Miniforge3-MacOSX-arm64.sh` or `Miniforge3-Linux-x86_64.sh`).

```bash
bash Miniforge3-MacOSX-arm64.sh  # or your platform-specific file
```

### Method 1: GitHub (Development Version)
```bash
git clone https://github.com/usnistgov/alignn
cd alignn
pip install -e .
```

### Method 2: PyPI Installation using `uv` (Recommended)

```bash
!pip install uv
!uv venv .venv --python 3.10
!source .venv/bin/activate
!pip install -q alignn
```

Ensure that you install the correct `dgl` version beforehand, compatible with your CUDA version. For example:

```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html  # For CUDA 12.1
```

For CPU-only systems:
```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html
```

---

## Examples

ALIGNN provides a suite of Google Colab notebooks showcasing various use cases:

| Task | Link | Description |
|------|------|-------------|
| Regression | [Notebook](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/alignn_jarvis_leaderboard.ipynb) | Single-output regression (e.g., exfoliation energy). |
| ML Force Field | [Notebook](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Train_ALIGNNFF_Mlearn.ipynb) | Train MLFF from scratch for Si. |
| Relaxation + Phonons | [Notebook](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Structure_Relaxation_Phonons_Interface.ipynb) | Use pre-trained ALIGNN-FF. |
| Scaling | [Notebook](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Timing_uMLFF.ipynb) | Timing analysis. |
| Amorphous Structures | [Notebook](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Fast_Melt_Quench.ipynb) | Melt-quench MD simulation. |
| Misc Tasks | [Notebook](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Training_ALIGNN_model_example.ipynb) | Training multiple regression/classification models. |

---

## Pretrained Models

Pre-trained ALIGNN and ALIGNN-FF models are hosted on [Figshare](https://figshare.com/projects/ALIGNN_models/126478). You can use them for fast inference with the provided scripts:

```bash
pretrained.py --model_name jv_formation_energy_peratom_alignn --file_format poscar --file_path POSCAR.vasp
```

For ALIGNN-FF:
```bash
run_alignn_ff.py --file_path POSCAR.vasp --task="optimize"
```

---

## Web Applications

1. [JARVIS-ALIGNN Web App](https://jarvis.nist.gov/jalignn/): Predict formation energy, total energy, bandgap.
2. [JARVIS-ALIGNN-FF App](https://jarvis.nist.gov/jalignnff/): Perform structure relaxations online.

---

## ALIGNN-FF & ASE Calculator

ALIGNN-FF integrates with ASE for tasks like structure optimization, energy-volume curves, and phonon calculations. A full script example is included in the documentation above.

---

## Performance Benchmarks

ALIGNN achieves state-of-the-art performance on JARVIS-DFT, Materials Project, QM9, and other datasets. See:
- [JARVIS-Leaderboard](https://pages.nist.gov/jarvis_leaderboard/)
- Specific tables in documentation for detailed metrics (classification/regression).

---

## Notes & Tips

- Install correct DGL-CUDA version for GPU support.
- Use batch size >32 for large datasets.
- `train_alignn.py` and `pretrained.py` are CLI tools.
- Pandas >1.2.3 required.
- Multi-GPU supported with `torchrun`.
- PyTorch Ignite dependency removed (as of March 2024).

---

## References

1. https://www.nature.com/articles/s41524-021-00650-1
2. https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00096b
3. Full publication list: https://jarvis-tools.readthedocs.io/en/master/publications.html

---

## Contribution Guide

Please see: [How to Contribute](https://github.com/usnistgov/jarvis/blob/master/Contribution.rst)

---

## Contact

- Report issues: [GitHub Issues](https://github.com/usnistgov/alignn/issues)
- Email: kamal.choudhary@nist.gov

---

## Funding

- [NIST-MGI](https://www.nist.gov/mgi)
- [NIST-CHIPS](https://www.nist.gov/chips)

---

## Code of Conduct

Please review: [Code of Conduct](https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md)


