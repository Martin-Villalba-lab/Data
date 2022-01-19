#library("Matrix")
library("tidyverse")
library("gridExtra")
library("mixtools")
options(repr.plot.width=8, repr.plot.height=8, repr.plot.res=200)

obs <- read_csv("computed/ifnagrko_obs.csv")
head(obs)

d <- read_csv("computed/hto_counts.csv.gz")
head(d)

hto <- d %>% select(-c(barcode, Unclear)) %>% as.matrix
rownames(hto) <- d$barcode
hto[1:5,]

k <- ncol(hto)
init <- rbind(diag(k), rep(1/k, k))
init

cats <- c(colnames(hto), "Unclear")
cats

hto.mix <- multmixEM(hto, theta=init)

obs$hto_ass <- cats[apply(hto.mix$posterior, 1, which.max)]
obs$hto_repl <- str_extract(obs$hto_ass, "(\\d)$")
obs$hto_zone <- str_extract(obs$hto_ass, "^[ROBSVZ]{3}")
obs$hto_sum <- rowSums(hto)

hto_post <- hto.mix$posterior
colnames(hto_post) <- cats

d.hto <- hto %>% as.data.frame %>%
    mutate(barcode=obs$barcode) %>%
    pivot_longer(-barcode, values_to="count", names_to="class") %>%
    left_join(hto_post %>% as.data.frame %>% mutate(barcode=obs$barcode) %>%
              pivot_longer(-barcode, values_to="posterior", names_to="class")) %>%
    left_join(select(obs, UMAP1, UMAP2, barcode, hto_sum, hto_ass))

options(repr.plot.width=13, repr.plot.height=6, repr.plot.res=200)
grid.arrange(
    d.hto %>% 
        ggplot(aes(count/hto_sum, hto_sum)) +
        geom_point(aes(colour=posterior)) +
        #geom_point() +
        scale_colour_distiller(type="div", palette=1, direction=1) +
        scale_y_log10() +
        scale_x_continuous(labels=scales::label_percent()) +
        labs(x="Fraction of total", y="Total hashtag UMIs") +
        #theme(legend.pos="bottom") +
        facet_wrap(~class),
    d.hto %>% 
        ggplot(aes(count/hto_sum, hto_sum)) +
        geom_point(colour="gray", data=filter(d.hto, hto_ass == "Unclear")) +
        geom_point(aes(colour=hto_ass), data=filter(d.hto, hto_ass != "Unclear")) +
        scale_colour_brewer(type="qual", palette="Paired") +
        scale_y_log10() +
        scale_x_continuous(labels=scales::label_percent()) +
        labs(x="Fraction of total", y="Total hashtag UMIs", colour="assignment") +
        #theme(legend.pos="none") +
        facet_wrap(~class),
    ncol=2
)

options(repr.plot.width=13, repr.plot.height=6)
grid.arrange(
    d.hto %>% 
        ggplot(aes(UMAP1, UMAP2)) +
        geom_point(aes(colour=posterior), size=0.1) +
        scale_colour_distiller(type="div", palette=1, direction=1) +
        coord_equal() +
        facet_wrap(~class),
    d.hto %>% 
        ggplot(aes(UMAP1, UMAP2)) +
        geom_point(colour="gray", size=0.1, data=filter(d.hto, hto_ass == "Unclear")) +
        geom_point(aes(colour=hto_ass), size=0.1, data=filter(d.hto, hto_ass != "Unclear")) +
        scale_colour_brewer(type="qual", palette="Paired") +
        coord_equal() +
        facet_wrap(~class),
    ncol=2
)

obs %>%
    select(barcode, hto_ass, hto_repl, hto_zone, hto_sum) %>%
    write_csv("computed/hto_assignment.csv")

sessionInfo()
