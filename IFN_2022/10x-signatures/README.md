# IFN signatures in 10x sequencing data old/young ko/wt

## How we got here

### 10x data

The 10x genomics `index-hopping-filter` tool was run on concatenated fastq files, treating each sequencing run as a different group. 
Filtered reads were quantified using kallisto|bustools (citation) provided by the `kb-python` (0.26.3) package. 
Sequencing runs were then combined in `scanpy` (1.6.0), filtered initially for empty droplets by keeping cells with at least 100 UMIs and genes with at least 3 cells.
Cells with less than 10% mitochondrial content and counts between 1000 and 20000, genes between 70 and 6000 which were also not called as doublets by `scrublet` (0.2.1) were kept thereafter.
Counts were normalized and log1p transformed using `scanpy` methods. 
Count matrices produced hereby were used to compute 50 dimensional PCAs, on which 15 neighbourhood graphs were built and UMAP embeddings computed.
The neighbourhood graph was also used to compute a leiden clustering. 
DBSCAN (provided by `scikit-learn` 0.23.2) was run on the UMAP embedding to assign ids to the various groups in the embedding. 
The DBSCAN cluster with highest `Pecam1` expression was determined to be endothelial cells and the cluster wight highest `Itgam` and `Ptprc` expression was determined to be Microglia.
The largest cluster was determined to be the neurogenic lineage.
Pseudotime was computed on the lineage cluster using `scanpy`'s diffusion pseudotime. 
The cell with the highest Aqp4 expression was used as the root cell for diffusion mapping (15D) which was then used with diffusion pseudotime using 10 diffusion components.
<!-- Lineage celltypes were assigned using `Seurat`'s label transfer methods. -->

