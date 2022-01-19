library("tidyverse")
library("DESeq2")
library("ggrepel")
library("ggrastr")
library("msigdbr")
#library("ggnewscale")
theme_set(theme_classic())

d.counts <- read_csv("data/star_counts.csv")

d.meta <- left_join(
    read_csv("data/star_mapping.csv"),
    read.csv("data/star_meta.csv") %>% dplyr::rename(id=X)
) %>% 
    mutate(treatment=ifelse(str_detect(id, "IFNb"), "IFNb", "ctrl"),
           repl=str_extract(id, "wt\\d+")) %>%
    as.data.frame
rownames(d.meta) <- d.meta$id
head(d.meta)

d.genes <- read_tsv("data/mart_export.txt") %>%
    dplyr::select(id=`Gene stable ID`, name=`Gene name`, description=`Gene description`)

d.counts.mat <- as.matrix(d.counts[2:ncol(d.counts)])
rownames(d.counts.mat) <- d.counts$gene
d.counts.mat[1:5,1:5]

options(repr.plot.width=6, repr.plot.height=6, repr.plot.res=200)
d.counts.mat %>% cor %>% heatmap

d.deseq.data <- DESeqDataSetFromMatrix(countData=d.counts.mat, colData=d.meta, design=~treatment)
d.deseq <- DESeq(d.deseq.data, test="Wald")#, reduced=~1)

options(repr.plot.width=6, repr.plot.height=6, repr.plot.res=200)
d.res <- results(d.deseq, altHypothesis="greater", contrast=c("treatment", "IFNb", "ctrl")) %>% as.data.frame %>% mutate(id=rownames(.)) %>% arrange(padj) %>% left_join(d.genes)
d.res %>%
    ggplot(aes(baseMean, log2FoldChange)) +
    geom_point(aes(colour="n.s."), data=filter(d.res, padj>=0.05), size=0.1, alpha=0.5) +
    geom_point(aes(colour="<5%"), data=filter(d.res, padj<0.05), size=0.1, alpha=0.5) +
    geom_text_repel(aes(colour="<5%", label=name), data=head(d.res, 50), max.overlaps=1000) +
    scale_colour_manual(values=c("n.s."="gray", "<5%"="firebrick3")) +
    labs(x="Reads", y="log2(fold change) IFNb vs Ctrl") +
    scale_x_log10() 

options(repr.plot.width=6, repr.plot.height=6, repr.plot.res=200)
d.res <- results(d.deseq, altHypothesis="greater", contrast=c("treatment", "IFNb", "ctrl")) %>% as.data.frame 
top.genes <- d.res %>% 
    arrange(padj) %>% 
    mutate(id=rownames(.)) %>%
    left_join(d.genes) %>%
    head(300)
d.res %>%
    ggplot(aes(baseMean, log2FoldChange)) +
    geom_point(colour="gray", data=d.res, size=0.1, alpha=0.5) +
    geom_point(colour="firebrick2", data=top.genes, size=0.1) +
    labs(x="Reads", y="log2(fold change) IFNb vs Ctrl") +
    scale_x_log10() 
writeLines(top.genes$name, "genesets/skabkin300.txt")

d.res %>%
    arrange(padj) %>%
    mutate(id=rownames(.)) %>%
    left_join(d.genes) %>%
    write_csv("DEBulkIFNb.csv")

options(repr.plot.width=6, repr.plot.height=6, repr.plot.res=200)
d.res <- results(d.deseq, altHypothesis="greater", contrast=c("treatment", "IFNb", "ctrl")) %>% as.data.frame 
top.genes <- d.res %>% 
    arrange(padj) %>% 
    mutate(id=rownames(.)) %>%
    left_join(d.genes) %>%
    head(300)
d.res %>%
    ggplot(aes(baseMean, log2FoldChange)) +
    geom_point_rast(raster.dpi=400, aes(colour="n.s."), data=filter(d.res, padj>=0.05), size=0.1, alpha=0.5) +
    geom_point_rast(raster.dpi=400, aes(colour="<5%"), data=filter(d.res, padj<0.05), size=0.5, alpha=0.5) +
    geom_point(aes(colour="selected"), data=top.genes, size=1) +
    #geom_text_repel(aes(colour="selected", label=name), data=head(top.genes, 20), max.overlaps=50, nudge_x=100, force=9) +
    scale_colour_manual(values=c("n.s."="gray", "<5%"="tomato", selected="firebrick4")) +
    labs(x="Reads", y="log2(fold change) IFNb vs Ctrl", title="NSC Type I Interferon Response") +
    scale_x_log10() 
dev.print(pdf, "plots/skabkin300_ma.pdf", width=6, height=6)
dev.print(svglite::svglite, "plots/skabkin300_ma.svg", width=6, height=6, fix_text_size=F)

d.entrez <- read_tsv("data/entrez_ensembl.txt") %>%
    dplyr::rename(id=`Gene stable ID`, name=`Gene name`, entrez=`NCBI gene (formerly Entrezgene) ID`)
d.entrez %>% head

d.wu <- read_csv("data/rice_isg_mouse.csv") %>%
    dplyr::select(wu_symbol=`Symbol`, entrez=`Entrez Gene ID`) %>% left_join(d.entrez)
#d.w
str(d.wu)
sum(is.na(d.wu$id))

writeLines(d.wu$name, "genesets/wu.txt")

length(intersect(d.wu$name, top.genes$name))

length(setdiff(d.wu$name, top.genes$name))

hallmarks <- c("HALLMARK_INFLAMMATORY_RESPONSE", "HALLMARK_INTERFERON_ALPHA_RESPONSE")
msig <- msigdbr::msigdbr(species="Mus musculus", category="H") %>% dplyr::filter(gs_name %in% hallmarks) 
hallmarks <- Map(function(name) dplyr::filter(msig, gs_name %in% name)$gene_symbol, hallmarks)

hallmarks

map2(hallmarks, paste0("genesets/", names(hallmarks), ".txt"), writeLines)

sessionInfo()
