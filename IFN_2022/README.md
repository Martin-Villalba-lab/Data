# Data from Carvajal et al.

# System requirements

- A computer capable of running python, R and julia. See the respective software for more detailed requirements.

This code was tested on a late 2021 Arch Linux.

# Expected install time

On a Linux system with conda and a reasonably fast internet connection it
should take less than 10 minutes to install the dependencies and download this
repository.

## Subdirectories

- `modelling`: Code for the population modelling results.
- `10x-signatures`: Code for the analysis of the 10x data.
- `riboseq`: Code for the analysis of the RiboSeq data.

## Files present

Code is kept in jupyter notebooks with julia, python and R kernels. 
These files include both the expected output as well as runtimes. For a compiled version with outputs see the HTML version of the files.
The source files are always the `*.ipynb` files with HTML and `.R`/`.py`/`.jl` generated for convenience along side them.

## Running this code

1. Download or clone this repository.
2. Either open the jupyter notebook files (`.ipynb`) and run those or open the other source files `.R`/`.py`/`.jl` and run the code within.

When files are named as `DD-name.ipynb` this means that
they are to be run in the order indicated by the digits in `DD`.
Otherwise the order doesn't matter.

## Installation instructions

Instructions are specific for each project and provided in the relevant sub-directories.
