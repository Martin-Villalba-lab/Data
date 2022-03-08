# IFN signatures in 10x sequencing data old/young ko/wt

## How we got here

### 10x data

The 10x genomics `index-hopping-filter` tool was run on concatenated fastq files, treating each sequencing run as a different group. 
Filtered reads were quantified using kallisto|bustools (citation) provided by the `kb-python` (0.26.3) package. 
Sequencing runs were then combined in `scanpy` (1.6.0), filtered initially for empty droplets by keeping cells with at least 100 UMIs and genes with at least 3 cells.

### RiboSeq RNA data

RiboSeq RNA libraries were trimmed using trim_galore 0.6.6 using default settings.
Trimmed reads were mapped and quantified against the GRCm38 version 102 genome from ENSEMBL using STAR 2.7.7a.
STAR output tables were concatenated in python with pandas.

## Installation instructions

It is advised to install software through conda.

1. Install R (`4.0.5` used here) & Bioconductor
2. Install python (`3.8.5` used here)
3. Install necessary R packages (see below)
4. Install necessary python packages (see below)

### Needed R packages as well as version used

- `tidyverse` (`1.3.1`)
- `DESeq2` (`1.30.1`)
- `ggrepel` (`0.9.1`)
- `ggrastr` (`0.2.3`)
- `msigdbr` (`7.4.1`)
- `gridExtra` (`2.3`)
- `mixtools` (`1.2.0`)
- `Matrix` (`1.3-4`)
- `ggnewscale` (`0.4.5`)

### Needed python packages as well as version used

- `anndata` (`0.7.6`)
- `scanpy` (`1.7.6`)
- `pandas` (`1.2.3`)
- `numpy` (`1.20.2`)
- `scrublet` (`0.2.1`)
- `scikit-learn` (`0.22.2`)
- `matplotlib` (`3.4.1`)
- `seaborn` (`0.11.1`)
- `plotnine` (`0.8.0`)

## How to run

1. Open a notebook
2. Run the notebook
