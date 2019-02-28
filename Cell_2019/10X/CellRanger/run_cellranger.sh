#!/bin/bash
## Call cell ranger with expected cell number and show pipeline overview on port 8080 

# Set number of expected cells
sudo /opt/10xGenomics/cellranger-2.0.0/cellranger count --id=Young10X --fastqs=/mnt/vol_1TB/10xworkingdir/fastqs/young/ --transcriptome=/opt/10xGenomics/refdata-cellranger-mm10-1.2.0/ --expect-cells=10000 --uiport=8080

sudo /opt/10xGenomics/cellranger-2.0.0/cellranger count --id=Old10X --fastqs=/mnt/vol_1TB/10xworkingdir/fastqs/old/ --transcriptome=/opt/10xGenomics/refdata-cellranger-mm10-1.2.0/ --expect-cells=10000 sudo /opt/10xGenomics/cellranger-2.0.0/cellranger count --id=Young10X --fastqs=/mnt/vol_1TB/10xworkingdir/fastqs/ --transcriptome=/opt/10xGenomics/refdata-cellranger-mm10-1.2.0/ --expect-cells=10000 sudo /opt/10xGenomics/cellranger-2.0.0/cellranger count --id=Young10X --fastqs=/mnt/vol_1TB/10xworkingdir/fastqs/ --transcriptome=/opt/10xGenomics/refdata-cellranger-mm10-1.2.0/ --expect-cells=10000 --uiport=8080
