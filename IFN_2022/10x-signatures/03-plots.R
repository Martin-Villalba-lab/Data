library("Matrix")
library("tidyverse")
library("ggnewscale")
library("ggrastr")
library("gridExtra")

options(repr.plot.width=8, repr.plot.height=8, repr.plot.res=200)

theme_set(theme_classic() + theme(strip.background=element_blank(), strip.text=element_text(hjust=0), axis.text.y=element_text(angle=90, hjust=0.5)))

mat <- readMM("computed/ifnagrko_raw_counts.mtx.gz")
obs <- read_csv("computed/ifnagrko_obs.csv") %>% 
    left_join(read_csv("computed/hto_assignment.csv")) %>% 
    mutate(celltype2=ifelse(celltype2 == "nan", NA, celltype2))
var <- read_csv("computed/ifnagrko_var.csv") %>% select(gene_name, gene_id)
rownames(mat) <- obs$barcode
colnames(mat) <- var$gene_name
mat[1:5,1:5]

sets.list <- Map(function(x) {readLines(str_glue("genesets/{x}.txt"))}, str_split(dir("genesets"), "\\.", simplify=T)[,1])

frac <- mat/rowSums(mat)

base_s <- 6
options(repr.plot.height=base_s, repr.plot.width=base_s*2)
type.long.order <- c("lineage:qNSC1", "lineage:qNSC2", "lineage:aNSC", "lineage:TAP", "lineage:NB", "neuron", "microglia", "endothelial")#, "oligo", "opc", "ob astrocyte", "ependymal")
type.order <- str_remove(type.long.order, "lineage:")
pos <- tibble(celltype=type.long.order, x=c(0, 1, 2, 3, 4, 5, 5.5, 6))
this.d <- names(sets.list) %>%
    mapply(function(this.name) { rowSums(log1p(1e6 * frac[,intersect(colnames(frac), sets.list[[this.name]])])) },  .) %>%
    as.data.frame %>%
    mutate(barcode=rownames(.)) %>%
    pivot_longer(-barcode, names_to="set", values_to="score") %>%
    left_join(obs) %>%
    filter(celltype %in% type.long.order, !is.na(hto_repl)) %>%
    group_by(set, age, genotype, celltype, hto_repl) %>%
    summarise(score=mean(score)) %>% 
    ungroup() %>%
    mutate(celltype_short=str_remove(celltype, "lineage:"),
           celltype_short=factor(celltype_short, type.order),
           celltype=factor(celltype, type.long.order),
           age=factor(age, c("young", "old")), 
           genotype=factor(genotype, c("WT", "KO")),
           set=case_when(str_detect(set, "HALLMARK") ~ set %>% str_replace_all("_", " ") %>% str_to_title(),
                         set == "wu" ~ "Wu et al. (2018)",
                         set == "skabkin300" ~ "NSC Type I Interferon Response")) %>%
    left_join(pos)
this.d %>%
    ggplot(aes(x, score, colour=age)) +
    facet_wrap(set~genotype, scales="free", ncol=4) +
    scale_colour_manual(values=c("old"="firebrick", "young"="cornflowerblue")) +
    scale_x_continuous(labels=str_remove(pos$celltype, "lineage:"), breaks=pos$x) + 
    labs(x="", y="Score") +
    theme(axis.text.x=element_text(angle=90, hjust=1, vjust=0.5),
          panel.grid.major.y=element_line(colour="gray"),
          panel.grid.minor.y=element_line(colour="gray", size=rel(0.5))) +
    geom_point(colour="transparent") +
    geom_point(data=filter(this.d, !str_detect(celltype, "lineage"))) +
    geom_line(aes(group=paste(hto_repl, age, genotype)), data=filter(this.d, str_detect(celltype, "lineage")))

base_s <- 6
options(repr.plot.height=base_s, repr.plot.width=base_s*1.7)
ps <- Map(function(this.name) {
    type.long.order <- c("lineage:qNSC1", "lineage:qNSC2", "lineage:aNSC", "lineage:TAP", "lineage:NB", "neuron", "microglia", "endothelial")#, "oligo", "opc", "ob astrocyte", "ependymal")
    #this.name <- "skabkin300"
    type.order <- str_remove(type.long.order, "lineage:")
    this.d <- rowSums(log1p(1e6 * frac[,intersect(colnames(frac), sets.list[[this.name]])])) %>%
        enframe(name="barcode", value="score") %>%
        left_join(obs, by="barcode") %>%
        #mutate(celltype_short=str_remove(celltype, "lineage:")) %>%
        filter(celltype %in% type.long.order, !is.na(hto_repl)) %>%
        group_by(age, genotype, celltype, hto_repl) %>%
        summarise(score=mean(score), .groups="drop") %>% 
        mutate(celltype_short=str_remove(celltype, "lineage:"),
               celltype_short=factor(celltype_short, type.order),
               celltype=factor(celltype, type.long.order),
               age=factor(age, c("young", "old")), 
               genotype=factor(genotype, c("WT", "KO")),
              )
    pos <- tibble(celltype=type.long.order, x=c(0, 1, 2, 3, 4, 5, 5.5, 6))
    this.name <- case_when(
        str_detect(this.name, "HALLMARK") ~ this.name %>% str_replace_all("_", " ") %>% str_to_title(),
        this.name == "wu" ~ "Wu et al. (2018)",
        this.name == "skabkin300" ~ "NSC Type I Interferon Response",
        T ~ this.name
    )
    this.d <- this.d %>%
        left_join(pos, by="celltype")
    this.d %>%
        ggplot(aes(x, score, colour=age, group=paste(genotype, age, hto_repl))) +
        geom_point(colour="transparent") +
        geom_line(data=filter(this.d, str_detect(celltype, "lineage:"))) +
        geom_point(data=filter(this.d, !str_detect(celltype, "lineage:")), position=position_dodge(width=0.1)) +
        scale_x_continuous(labels=str_remove(pos$celltype, "lineage:"), breaks=pos$x) + 
        #scale_x_discrete(breaks=1:length(type.order), labels=type.order) +
        scale_colour_manual(values=c("young"="cornflowerblue", "old"="firebrick")) +
        labs(title=this.name, x="", y="Score", colour="Age") +
        theme(axis.text.x=element_text(angle=90, hjust=1, vjust=0.5),
              panel.grid.major.y=element_line(colour="gray"),
              panel.grid.minor.y=element_line(colour="gray", size=rel(0.5))) +
        facet_wrap(~genotype)
}, names(sets.list))
ps$ncol <- 2
do.call(grid.arrange, ps)
dev.print(pdf, "plots/signatures.pdf", height=6, width=6*1.7)
dev.print(svglite::svglite, "plots/signatures.svg", width=6*1.7, height=6, fix_text_size=F)



base_s <- 6
options(repr.plot.height=base_s, repr.plot.width=8)
type.order <- c("qNSC1", "qNSC2", "aNSC", "TAP", "NB", "microglia", "endothelial", "neuron")
this.cols <- c(hcl.colors(5, "Blue-Red")[1:5], "goldenrod2", "forestgreen", "indianred")
this.name <- "skabkin300"
this.d <- rowSums(log1p(1e6 * frac[,intersect(colnames(frac), sets.list[[this.name]])])) %>%
    enframe(name="barcode", value="score") %>%
    left_join(obs) %>%
    mutate(celltype=str_remove(celltype, "lineage:")) %>%
    filter(celltype %in% type.order) %>%
    mutate(celltype=factor(celltype, type.order))
names(this.cols) <- type.order
this.d.lin <-  this.d %>%
    filter(!is.na(celltype2)) %>%
    group_by(celltype) %>%
    summarise(UMAP1=median(UMAP1), UMAP2=median(UMAP2))
this.d.off <-  this.d %>%
    filter(is.na(celltype2)) %>%
    group_by(dbscan) %>%
    summarise(UMAP1=median(UMAP1), UMAP2=median(UMAP2), celltype=celltype[1])
this.d %>%
    ggplot(aes(UMAP1, UMAP2)) +
    geom_point_rast(aes(colour=celltype), size=0.1, data=this.d, raster.dpi=400) +
    geom_label(aes(label=celltype), label.r=unit(0, "mm"), nudge_y=0, data=this.d.lin) +
    geom_label(aes(label=celltype), label.r=unit(0, "mm"), nudge_y=2, data=filter(this.d.off, UMAP2>0)) +
    geom_label(aes(label=celltype), label.r=unit(0, "mm"), nudge_y=-1, data=filter(this.d.off, UMAP2<0)) +
    scale_colour_manual(values=this.cols) +
    theme_void() +
    theme(legend.pos="none") +
    coord_equal()
dev.print(pdf, "plots/celltypes.pdf", height=6, width=6)
dev.print(svglite::svglite, "plots/celltypes.svg", width=6, height=6, fix_text_size=F)

options(repr.plot.width=6, repr.plot.height=4)
cols <- data.frame(celltype2=c("qNSC1", "qNSC2", "aNSC", "TAP", "NB"),
                   col=hcl.colors(5, "Blue-Red")[1:5])
labs <- obs %>%
    filter(!is.na(celltype2)) %>%
    group_by(celltype2) %>%
    summarise(dpt_pseudotime=mean(range(dpt_pseudotime))) %>%
    left_join(cols)
obs %>%
    #mutate(dpt_pseudotime=rank(dpt_pseudotime)) %>%
    left_join(enframe(mat[,"EYFP"], name="barcode", value="expression")) %>%
    filter(!is.na(celltype2), expression>0) %>%
    mutate(celltype2=factor(celltype2, type.order)) %>%
    ggplot(aes(dpt_pseudotime)) +
    scale_colour_manual(values=c("WT young"="cornflowerblue", "WT old"="deepskyblue4", "KO young"="indianred1", "KO old"="firebrick")) +
    labs(title="YFP+ cells", colour="Group", y="Count", x="Pseudotime") +
    geom_density(aes(y=after_stat(count), colour=paste(genotype, age))) +
    new_scale_colour() +
    geom_rug(aes(colour=celltype2), sides="b", alpha=1) +
    labs(colour="Celltype") +
    scale_colour_manual(breaks=labs$celltype2, values=labs$col, guide=F) +
    scale_x_continuous(breaks=labs$dpt_pseudotime, labels=labs$celltype2)
dev.print(pdf, "plots/pseudotime_yfp.pdf", height=4, width=6)
dev.print(svglite::svglite, "plots/pseudotime_yfp.svg", width=6, height=4, fix_text_size=F)

colnames(mat)[startsWith(colnames(mat), "Ifn")]

options(repr.plot.width=12, repr.plot.height=8)
ps <- log1p(1e6*frac[,c("Ifnar1", "Ifnar2", "Ifngr1", "Ifngr2")]) %>%
    as.matrix %>%
    as.data.frame %>%
    mutate(barcode=rownames(.)) %>%
    pivot_longer(-barcode) %>%
    left_join(obs) %>%
    mutate(celltype=str_remove(celltype, "lineage:")) %>%
    #filter(celltype %in% type.order) %>%
    mutate(celltype=factor(celltype, type.order)) %>%
    #filter(genotype == "WT") %>%
    mutate(group=paste(genotype, age)) %>%
    nest(-group) %>%
    arrange(group) %>%
    deframe %>%
    #map2(., names(.), ~ggplot(.x, aes(dpt_pseudotime, value, colour=celltype)) + geom_point_rast(alpha=0.2, raster.dpi=400) + geom_smooth(colour="black", se=F) + scale_colour_manual(values=this.cols[1:5]) + labs(y="Expression", x="Pseudotime", colour="Celltype", title=.y) + facet_wrap(~name) + guides(colour=guide_legend(override.aes=list(alpha=1))))
    map2(., names(.), ~ggplot(.x, aes(dpt_pseudotime, value, colour=celltype)) + geom_point_rast(alpha=0.2, raster.dpi=400)  + scale_colour_manual(values=this.cols[1:5]) + labs(y="Expression", x="Pseudotime", colour="Celltype", title=.y) + facet_wrap(~name) + guides(colour=guide_legend(override.aes=list(alpha=1))))
do.call(grid.arrange, ps)
dev.print(pdf, "plots/pseudotime_receptors.pdf", height=8, width=12)
dev.print(svglite::svglite, "plots/pseudotime_receptors.svg", width=12, height=8, fix_text_size=F)



options(repr.plot.width=6, repr.plot.height=6)
log1p(1e6*frac[,c("Ifnar1", "Ifnar2", "Ifngr1", "Ifngr2")]) %>%
    as.matrix %>%
    as.data.frame %>%
    mutate(barcode=rownames(.)) %>%
    pivot_longer(-barcode) %>%
    mutate(pos=value>0) %>%
    left_join(obs) %>%
    ggplot(aes(dpt_pseudotime, as.numeric(pos))) +
    geom_jitter(alpha=0.1, width=0, height=0.2) +
    geom_smooth() +
    facet_wrap(~name)

options(repr.plot.width=9, repr.plot.height=9)
log1p(1e6*frac[,c("Ifnar1", "Ifnar2", "Ifngr1", "Ifngr2")]) %>%
    as.matrix %>%
    as.data.frame %>%
    mutate(barcode=rownames(.)) %>%
    pivot_longer(-barcode) %>%
    left_join(obs) %>%
    ggplot(aes(UMAP1, UMAP2, colour=value)) +
    coord_fixed() +
    geom_point(size=0.1) +
    facet_wrap(~name)

mat[1:9,1:9]



sessionInfo()
