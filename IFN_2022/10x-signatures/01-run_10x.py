#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import os
import functools
import gzip
import math
import json
import anndata
import scipy.io
import scipy.sparse
import pandas as pd
import scanpy as sc
import numpy as np
import scrublet
import sklearn.cluster
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *


# # Config

# In[2]:


theme_set(theme_classic() + theme(panel_grid_major=element_line()))


# In[3]:


rules = {
    "frac_mito": (0, 0.1),
    "counts": (1000, 20000),
    "genes": (70, 6000),
}


# In[4]:


celltype_map = {
    "neuron": ["Rbfox3", "Calb2", "Nefl", "Th"],
    "lineage": ["Aqp4", "Egfr", "Mki67", "Dcx"],
    "microglia": ["Itgam", "Ptprc"],
    "oligo": ["Cldn11"],
    "oec": ["Apod"],
    "pericyte": ["Foxd1", "Colec12"],
    "opc": ["Olig1", "Olig2", "Lhfpl3", "Cntn1"],
    "endothelial": ["Prom1", "Cldn5", "Pecam1"],
    "ependymal": ["Cald1", "Sod3"],
}


# In[5]:


hto_order = ["SVZ1", "SVZ2", "ROB1", "ROB2", "Unclear"]


# # Load data

# In[6]:


d = sc.read("data/ifnagrko_raw.h5ad")
d.var_names_make_unique()
d


# # Preprocessing

# ## Filter

# In[7]:


sc.pp.filter_cells(d, min_counts=100)
sc.pp.filter_genes(d, min_cells=3)
d


# In[8]:


d.var["mt"] = d.var_names.str.startswith("mt-")
d.obs["counts"] = d.X.sum(axis=1).A1
d.obs["genes"] = (d.X > 0).sum(axis=1).A1
d.obs["counts_mito"] = d[:,d.var.mt].X.sum(axis=1).A1
d.obs["frac_mito"] = d.obs["counts_mito"] / d.obs["counts"]
d


# In[9]:


scrub = scrublet.Scrublet(d.X)
d.obs["scrub_score"], d.obs["scrub_label"] = scrub.scrub_doublets()
d


# In[10]:


exclusions = {key: ~d.obs[key].between(*rules[key], inclusive=True) for key in ["frac_mito", "counts", "genes"]}
exclusions["scrublet"] = d.obs.scrub_label
d.obs["exclude"] = functools.reduce(lambda a, b: a | b, exclusions.values())


# In[11]:


d.obs.exclude.value_counts()


# In[12]:


d = d[~d.obs.exclude,:]


# In[13]:


d


# ## Normalisation

# In[14]:


d.layers["raw_counts"] = d.X.copy()


# In[15]:


sc.pp.normalize_total(d, target_sum=1e4)
sc.pp.log1p(d)
d.layers["lognormal"] = d.X.copy()
d


# # Sample names

# In[16]:


d.obs["genotype"] = [x.split("-")[1].split("_")[0] for x in d.obs.index]
d.obs["age"] = ["old" if "20mo" in x else "young" for x in d.obs.index]


# In[17]:


d.obs.age


# In[18]:


{x.split("-")[1].split("_")[1] for x in d.obs.index}


# In[19]:


d.obs.genotype


# # Dimensionality reductions

# In[20]:


sc.pp.scale(d, zero_center=False)
sc.tl.pca(d, n_comps=50)
sc.pp.neighbors(d, n_neighbors=15)
sc.tl.umap(d)
d


# In[21]:


plt.rcParams['figure.figsize'] = [5, 5]
plt.rcParams['figure.dpi'] = 150
sc.pl.umap(d)


# # Cluster

# In[22]:


sc.tl.leiden(d)
d.obs["dbscan"] = sklearn.cluster.DBSCAN().fit_predict(d.obsm["X_umap"]).astype(str)
dbscan_value_counts = d.obs.dbscan.value_counts()
lineage_cluster_id = dbscan_value_counts.index[dbscan_value_counts.argmax()]
d.obs["is_lineage"] = (d.obs.dbscan == lineage_cluster_id)
d


# In[23]:


plt.rcParams['figure.figsize'] = [3, 3]
plt.rcParams['figure.dpi'] = 150
sc.pl.umap(d, color=["dbscan", "leiden", "Aqp4", "Dcx", "Pecam1", "Cldn5", "Itgam", "Ptprc"], ncols=2)


# # HTO zone assignment

# In[24]:


rob_htos = [h.startswith("ROB") for h in hto_order]
svz_htos = [h.startswith("SVZ") for h in hto_order]
d.obs["rob_sum"] = d.obsm["hto"][:,rob_htos].sum(axis=1)
d.obs["svz_sum"] = d.obsm["hto"][:,svz_htos].sum(axis=1)
d.obs["hto_sum"] = d.obs.rob_sum + d.obs.svz_sum
d.obs["svz_frac"] = d.obs.svz_sum / d.obs.hto_sum
d


# In[25]:


plt.rcParams['figure.figsize'] = [4, 4]
plt.rcParams['figure.dpi'] = 150
sc.pl.umap(d, color=["svz_frac", "leiden", "dbscan"], color_map=matplotlib.cm.RdBu)


# # Celltypes

# In[26]:


marker_map_tups = functools.reduce(lambda x, y: x+y, [[(key, value) for value in markers] for (key, markers) in celltype_map.items()])
marker_genes = [tup[1] for tup in marker_map_tups]
expr = pd.DataFrame(d[:,marker_genes].X.todense(), index=d.obs.index, columns=pd.MultiIndex.from_tuples(marker_map_tups))
expr = expr.groupby(level=0, axis=1).sum()


# In[27]:


mat = d.obs[["dbscan"]].join(expr, how="left").groupby("dbscan").agg("median")
sns.heatmap(mat.divide(mat.sum(axis=1), axis=0))


# In[28]:


typemat = (np.max(mat, axis=1) == mat.transpose()).transpose()
sns.heatmap(typemat)


# In[29]:


typemap = typemat.apply(lambda d: d.index[d][0], axis=1).to_dict()
typemap["-1"] = "singleton"
typemap


# In[30]:


d.obs["celltype1"] = [typemap[i] for i in d.obs.dbscan]


# In[31]:


plt.rcParams['figure.figsize'] = [4, 4]
plt.rcParams['figure.dpi'] = 150
sc.pl.umap(d, color=["svz_frac", "celltype1", "dbscan", "leiden"], color_map=matplotlib.cm.RdBu, ncols=2)


# # Celltype refinement
# 
# - `leiden:19` and `leiden:20` are actually not SVZ cells and thus should not be called `lineage`.
# - `leiden:22` is likely to be further doublets

# In[32]:


sc.pl.umap(d, color=["scrub_score"])


# In[33]:


d.obs.celltype1 = d.obs.celltype1.astype(str)
d.obs.loc[d.obs.leiden=="19", "celltype1"] = "ob astrocyte"
d.obs.loc[d.obs.leiden=="20", "celltype1"] = "ob astrocyte"
d.obs.loc[(d.obs.leiden=="22"), "celltype1"] = "doublet"


# In[34]:


sc.pl.umap(d, color=["celltype1"])


# # Pseudotime

# In[35]:


dlin = d[d.obs.celltype1 == "lineage"].copy()
dlin.uns["iroot"] = np.argmax(dlin[:,"Aqp4"].X)
sc.tl.pca(dlin)
sc.pp.neighbors(dlin)
sc.tl.diffmap(dlin)
sc.tl.dpt(dlin)
d.obs = d.obs.join(dlin.obs[["dpt_pseudotime"]], how="left")
del dlin
d


# In[36]:


sc.pl.umap(d, color=["dpt_pseudotime"])


# ## Lineage celltypes

# In[37]:


markers = ["Aqp4", "Egfr", "Dcx", "Mki67", "Gria1", "S100b"]


# In[38]:


bin_count = 100


# In[ ]:





# In[39]:


plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 150
lineage_cells = d.obs.celltype1 == "lineage"
marker_data = pd.DataFrame(d[lineage_cells,markers].layers["lognormal"].todense(), columns=markers, index=d[lineage_cells,:].obs.index).join(d.obs, how="left")
marker_data["bin"] = pd.cut(marker_data.dpt_pseudotime, bins=bin_count)
marker_mat = marker_data.reset_index()[["bin"] + markers].groupby(["bin"]).sum()
sns.heatmap(marker_mat.divide(marker_mat.sum(axis=1), axis=0).transpose())


# In[40]:


cuts = [0.13, 0.31, 0.73, 0.842]
types = pd.DataFrame(
    {"lo": [0] + cuts,
     "hi": cuts + [1],
     "type": ["qNSC1", "qNSC2", "aNSC", "TAP", "NB"]
    }
)
types["dpt_pseudotime"] = types.lo + (types.hi-types.lo)/2
marker_mat = marker_data.reset_index()[["bin"] + markers].groupby("bin").sum()
marker_mat = marker_mat.divide(marker_mat.sum(axis=1), axis=0)
marker_mat = marker_mat.reset_index()
marker_mat["dpt_pseudotime"] = [b.mid for b in marker_mat.bin]
gg = (
    ggplot(marker_mat.melt(id_vars=["dpt_pseudotime", "bin"]), aes("dpt_pseudotime", "value")) +
    geom_bar(aes(fill="variable"), stat="identity", width=1/bin_count) +
    geom_vline(xintercept=cuts) +
    geom_label(aes(x="dpt_pseudotime", y=.99, label="type"), data=types, size=8.5, va="top", label_r=0, label_size=0) +
    coord_cartesian(expand=False) +
    labs(y="Relative expression", x="Binned Pseudotime", colour="Gene")
)
ggsave(gg, "plots/celltype_calling.pdf")
ggsave(gg, "plots/celltype_calling.svg")
gg


# In[41]:


ggplot(d.obs, aes("dpt_pseudotime")) + geom_histogram(bins=100) + coord_cartesian(expand=False)


# In[42]:


d.obs["celltype2"] = [
    np.nan if pd.isna(x) else
    "qNSC1" if x < cuts[0] else 
    "qNSC2" if x < cuts[1] else 
    "aNSC"  if x < cuts[2] else
    "TAP"   if x < cuts[3] else
    "NB"    
    for x in d.obs.dpt_pseudotime
]
(
    ggplot(d.obs.assign(UMAP1 = d.obsm["X_umap"][:,0], UMAP2=d.obsm["X_umap"][:,1]), aes("UMAP1", "UMAP2")) +
    geom_point(aes(colour="celltype2"), size=0.1) +
    theme_void() +
    coord_equal()
)


# In[43]:


d.obs["celltype"] = [c1 if pd.isna(c2) else f"{c1}:{c2}" for (c1, c2) in zip(d.obs.celltype1, d.obs.celltype2)]


# # Summary

# In[44]:


ddata = d.obs.assign(UMAP1 = d.obsm["X_umap"][:,0], UMAP2=d.obsm["X_umap"][:,1])


# In[45]:


(
    ggplot(ddata, aes("UMAP1", "UMAP2")) +
    geom_point(colour="gray", size=0.1, data=ddata.loc[ddata.celltype1 != "lineage"]) +
    geom_point(aes(colour="dpt_pseudotime"), size=0.1, data=ddata.loc[ddata.celltype1 == "lineage"]) +
    theme(axis_line=element_blank(), axis_ticks=element_blank(), panel_grid=element_blank(), axis_text=element_blank(), axis_title=element_blank()) +
    #theme_void() +
    #theme(legend_position="none") +
    coord_equal()
)


# In[46]:


(
    ggplot(ddata, aes("UMAP1", "UMAP2")) +
    geom_point(aes(colour="svz_frac"), size=0.1) +
    scale_colour_cmap(cmap_name="RdBu") +
    theme(axis_line=element_blank(), axis_ticks=element_blank(), panel_grid=element_blank(), axis_text=element_blank(), axis_title=element_blank()) +
    #theme_void() +
    #theme(legend_position="none") +
    coord_equal()
)


# In[47]:


dlabel1 = d.obs.assign(UMAP1 = d.obsm["X_umap"][:,0], UMAP2=d.obsm["X_umap"][:,1]).loc[:,["celltype1", "UMAP1", "UMAP2"]].groupby("celltype1").median()
dlabel1 = dlabel1.reset_index().loc[dlabel1.index != "lineage"]
dlabel2 = d.obs.assign(UMAP1 = d.obsm["X_umap"][:,0], UMAP2=d.obsm["X_umap"][:,1]).loc[d.obs.celltype1 == "lineage",["celltype2", "UMAP1", "UMAP2"]].groupby("celltype2").median().reset_index()
(
    ggplot(ddata, aes("UMAP1", "UMAP2")) +
    geom_point(colour="gray", size=0.1, data=ddata.loc[ddata.celltype1 != "lineage"]) +
    geom_point(aes(colour="celltype2"), size=0.1, data=ddata.loc[ddata.celltype1 == "lineage"]) +
    geom_label(aes(label="celltype1"), data=dlabel1[dlabel1.UMAP1 > 5], nudge_x=3, nudge_y=1, alpha=0.6) +
    geom_label(aes(label="celltype1"), data=dlabel1[dlabel1.UMAP1 < 5], nudge_x=-3, nudge_y=0, alpha=0.6) +
    geom_label(aes(label="celltype2", colour="celltype2"), data=dlabel2, nudge_x=0, alpha=1.0) +
    theme_void() +
    theme(legend_position="none") +
    coord_equal()
)


# In[48]:


plt.rcParams['figure.figsize'] = [3, 3]
plt.rcParams['figure.dpi'] = 150
sc.pl.umap(d, color=["Pecam1", "Itgam", "Ptprc", "Aqp4", "S100b", "celltype"], ncols=3)


# # Export

# In[49]:


d.write_h5ad("computed/ifnagrko.h5ad")


# In[50]:


d.obs.celltype.value_counts()


# In[51]:


columns = ["celltype", "celltype1", "celltype2", "dpt_pseudotime", "svz_frac", "hto_sum", "leiden", "dbscan", "frac_mito", "counts", "genes", "scrub_score", "age", "genotype"]


# In[52]:


d.obs[columns].assign(UMAP1=d.obsm["X_umap"][:,0], UMAP2=d.obsm["X_umap"][:,1]).to_csv("computed/ifnagrko_obs.csv")


# In[53]:


d.var.reset_index()[["gene_name", "gene_id"]].to_csv("computed/ifnagrko_var.csv")


# In[54]:


with gzip.open("computed/ifnagrko_raw_counts.mtx.gz", "wb") as f:
    scipy.io.mmwrite(f, d.layers["raw_counts"])


# In[55]:


with gzip.open("computed/ifnagrko_log_norm.mtx.gz", "wb") as f:
    scipy.io.mmwrite(f, d.layers["lognormal"])


# In[56]:


pd.DataFrame(d.obsm["hto"], columns=hto_order, index=d.obs.index).to_csv("computed/hto_counts.csv.gz")


# # Info

# In[57]:


get_ipython().system('pip list')

