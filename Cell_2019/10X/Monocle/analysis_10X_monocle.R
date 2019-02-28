# Pseudotime assignment

require(Matrix)
require(Biobase)
path <- '../count_table/filtered_gene_bc_matrices_mex/mm10'
setwd(path)
umi_Matrix <- readMM('matrix.mtx')

genes <- read.table(file = "genes.tsv", sep = "\t")
symbols <- as.character(genes$V2)
ensembls <- as.character(genes$V1)
fd <- data.frame(Genes = symbols, row.names = ensembls)

cells <- read.table(file = "barcodes.tsv", sep = "\t")

setwd("~/Downloads")
require(ggplot2)
require(gridExtra)

barcodes <- read.csv(file = "~/Downloads/10X_2_metadata.csv")

PreCluster <- as.character( barcodes$celltype )
names(PreCluster) <- barcodes$X

barcodes <- as.character(cells$V1)

Type <- rep('', times = length(barcodes))
names(Type) <- barcodes
Type[names(PreCluster)] <- PreCluster

Age <- rep('old', times = length(barcodes))
Age[grep('-2', barcodes)] <- 'young'

cells <- data.frame(Barcode = barcodes, Type = Type, Age = Age, row.names = barcodes)

yc <- cells[cells$Age == 'young', ]

bar <- as.character(yc$Barcode)
barMatch <- vapply(bar, function(x){  paste0( unlist(strsplit(x, split = '-'))[1], '-1' ) }, 'AAACCTGAGGAATCGC-2' )

celltype <- as.character(yc$Type)
names(celltype) <- barMatch

celltype <- celltype[filtered@sample]


require(monocle)
ix <- which(!is.element(Type, c('OPC', 'OD', '')))
NSC_ <- newCellDataSet(as(umi_Matrix, "sparseMatrix")[ ,ix],
                       phenoData = new("AnnotatedDataFrame", data = cells[ix, ]),
                       featureData = new("AnnotatedDataFrame", data = fd),
                       lowerDetectionLimit = 0.5,
                       expressionFamily = negbinomial.size())

# save(NSCM, file = 'NSCM.RData')

NSC_ <- estimateSizeFactors(NSC_)
NSC_ <- estimateDispersions(NSC_)
NSC_ <- detectGenes(NSC_, min_expr = 0.1)
disp_table <- dispersionTable(NSC_)
ordering_genes <- subset(disp_table, mean_expression >= 0.6 & dispersion_empirical >= 3 * dispersion_fit)$gene_id

NSC_ <- setOrderingFilter(NSC_, ordering_genes)
plot_ordering_genes(NSC_)
NSC_ <- reduceDimension(NSC_, max_components=2)
NSC_ <- orderCells(NSC_)
p1 <- plot_cell_trajectory(NSC_, color_by = "Pseudotime")

plot_cell_trajectory(NSC_, color_by = "State")
