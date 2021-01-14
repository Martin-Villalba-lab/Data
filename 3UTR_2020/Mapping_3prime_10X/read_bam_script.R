### read-in 3'UTR ends from 10X bam files
suppressPackageStartupMessages(library(BiocParallel))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(require(stringr))
suppressPackageStartupMessages(library(GenomicAlignments))
suppressPackageStartupMessages(library(Rsamtools))

read_in_10Xutr <- function(dir, batch){
  # read in all BAM-Files in directory
  files <- list.files(dir)
  files <- files[grep('^RUN_', files)]
  bams <- paste0( dir, files )
  valid <- sapply(bams, function(x){ any(list.files(x) == 'outs') })
  table(valid)
  bams <- bams[valid]
  message(length(bams))
  print(bams)
  #paral          <- SnowParam(workers = 15, type = "SOCK", progressbar = TRUE)
  #mapping_list <- bplapply(bams, extract3utrPositions10X)
mapping_list <- lapply(bams, extract3utrPositions10X)
  names(mapping_list) <- paste0(batch, '_', bams)
  return(mapping_list)
}


extract3utrPositions10X <- function(pathtobam, chunk_size = 250){
  # extract the positions of 3'UTR regions
  load(file = 'threeUTR_human.rda')
  pathtobam <- paste0(pathtobam, '/outs/possorted_genome_bam.bam')
  utr_tbl_master <- list()
  nregions       <- seq_along(threeUTR_human)
  chunks         <- split( nregions, ceiling(seq_along(nregions)/chunk_size)) 
  n_chunks       <- seq_along(chunks)
  paral          <- SnowParam(workers = 16, type = "SOCK", progressbar = TRUE)
  message(paste('Started processing chunks!!', Sys.time(), sep = '  ') )
  
  #for(n in n_chunks[1:3]){
    # for every chunk
   # chunk_idx <- chunks[[n]]
    #message(paste0('Processing chunk: ', n, ' of ', length(n_chunks) ))
    
    utr_tbl_master <- bplapply(n_chunks,
                               function(x,
                                        chunks,
                                        threeUTR_human,
                                        pathtobam){ # parallel loop start
                                 # for every chunk
                                 chunk_idx <- chunks[[x]]
                                 if( x %% 10 == 0) message(paste0('Processing chunk: ', x, ' of ', length(n_chunks) ))
                                 
    # define the paramters for the BAM-File
    param <- ScanBamParam(flag=scanBamFlag(
      isDuplicate=FALSE,
      isSecondaryAlignment=FALSE),
     mapqFilter = 10,
     which = threeUTR_human[chunk_idx],
     what = c('pos', 'qname', 'seq' ), tag = c('CB', 'UB'))
    
    # load the alignments into memory
    yngal <- readGAlignments(pathtobam,
                             param=param,
                             use.names = TRUE,
                             with.which_label=TRUE)
    
    # extract the mapping position of read1
    read1          <- yngal #second(yngal) #<- first(yngal)
    # get the orientation of read1
    read1_strand   <- decode(strand(read1))
    # get the assignment of regions as full vector
    read1_region   <- decode(read1@elementMetadata@listData$which_label)
    # make a list with every entry being a region
    read1_by_range <- tapply( seq_along(read1), read1_region, function(x, y){ y[x] }, y = read1 )
    # exclude those regions with 0 entries
    read_idx       <- !unlist(lapply(read1_by_range, is.null))
    # get the starts of 3'UTR regions                         ... keep strand orientation
    region_starts  <- start(threeUTR_human[chunk_idx][read_idx], ignore.stand = FALSE) 
    region_ends    <- end(threeUTR_human[chunk_idx][read_idx],   ignore.stand = FALSE) 
    # compute the local 3'UTR positions (3'UTR lengths, considering the region and strand orientation)
    local_strand   <- strand(threeUTR_human)[chunk_idx][read_idx] %>% decode() %>% as.character()
    read1_pos      <- mapply(function(x, y1, y2, z){ if(z == '+') return(end(x) - y1); return(y2 - start(x)) },    # return(end(x) - y1); return(y2 - start(x)) }
                             x = read1_by_range[read_idx], y1 = region_starts, y2 = region_ends, z = local_strand )
    # get the read-name, read orientation & read1 sequence
    read1_cig      <- lapply(read1_by_range[read_idx], function(x){ cigar(x) } )
    read1_ori      <- lapply(read1_by_range[read_idx], function(x){ decode(strand(x)) } )
    read1_seq      <- lapply(read1_by_range[read_idx], function(x){  x@elementMetadata@listData$seq  } )
    read1_ub      <- lapply(read1_by_range[read_idx], function(x){  x@elementMetadata@listData$UB  } )
    read1_cb      <- lapply(read1_by_range[read_idx], function(x){  x@elementMetadata@listData$CB  } )
    # compute the reverse complement if on the '-' strand
    read1_rev      <- mapply(function(x, y){ x[y == '-'] <- reverseComplement(x[y == '-']); x }, x = read1_seq, y = read1_ori )
    # extract barcodes and read lengths
    read1_len      <- lapply(read1_rev, function(x) { str_length(x) } )
    # write this into a tibble
    utr_tbl_list   <- mapply(function(x1, x3, x4, x5, x6, x7){ 
      tibble(utr_len = x1, strand = x3, cigar = x4, trim_len = x5, read1_cb = x6, read1_ub = x7)
    },
    x1 = read1_pos,
    x3 = read1_ori,
    x4 = read1_cig,
    x5 = read1_len,
    x6 = read1_cb,
    x7 = read1_ub,
    SIMPLIFY = FALSE)
    
    # include the UMI correction
    utr_tbl_list  <- lapply(utr_tbl_list, function(x){
                     out <- tapply( x$utr_len, paste0(x$read1_cb, '_', x$read1_ub), median )
                     names(out) <- substring(names(out), 1, 18)
                     return(out) })

    # name the lists by the regions
    gene_chunk         <- names(threeUTR_human)[chunk_idx][read_idx]
    names(utr_tbl_list) <- gene_chunk
    # exclude 'spliced' alignments by the region length
    local_length   <- width(threeUTR_human)[chunk_idx][read_idx] %>% decode()
    utr_tbl_list   <- mapply(function(x, y){ x[which(x < y)] }, x = utr_tbl_list, y = (local_length + 100) )
    utr_tbl_list   <- utr_tbl_list[sapply(utr_tbl_list, function(x){ length(x) > 1 })]
    #
    utr_tbl_list
    }, chunks, threeUTR_human, pathtobam)#
    
    utr_tbl_master <- unlist(utr_tbl_master, recursive = FALSE)
    utr_tbl_master <- utr_tbl_master[ !is.na(names(utr_tbl_master)) ]
    message(paste('All chunks finished!', Sys.time(), sep = '  ') )
  return(utr_tbl_master)
}
###