# LineSeeker

# Validate.py

This script checks whether a laptop or cluster has issues running Numba properly, whether CPU resources are being restricted, and whether code sections expected to run in parallel are actually doing so.

# Environment setup

This repository should ship an environment definition, not a copied Conda environment directory.

CPU:

```bash
conda env create -f environment.yml
conda activate lineseeker
```

GPU:

```bash
conda env create -f environment-gpu.yml
conda activate lineseeker-gpu
```

# How to use it!

just run these commands in order:

Searchline: <code>python SearchLine.py -Cube PATH_TO_FITS -MinSN N -MaxSigmas N -backend '' -NSigmaSpatial N</code>.

GetLines: <code>python getline.py -Cube PATH_TO_FITS -MinSN N -MaxSigmas N -backend CPU -NSigmaSpatial N</code>.
