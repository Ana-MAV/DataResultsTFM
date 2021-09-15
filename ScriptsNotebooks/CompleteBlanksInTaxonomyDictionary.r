# When there is only one taxonomy level known.
uniqueKnownTaxaLevel="Class" # Put here the level of the taxonomy of the values given.
taxmat = matrix(nrow=ntaxa(OTU),ncol=7)
rownames(taxmat) <- taxa_names(OTU)
colnames(taxmat) <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
# To complete the remainder parent ranks of the taxonomy classification
# To parser some of the hand-written class rank taxonomy, to make possible to find their parent taxonomy automatically
taxmat[,uniqueKnownTaxaLevel] <- unlist(lapply(rownames(taxmat),function(y) gsub("*_no_class","\\",y)))
library(taxize)
for (i in 1:nrow(taxmat)){
  # With taxmat()
  # rows=1: To select the first result if there are several ones. ask=FALSE, not interactive for selecting in results.
  if(!is.na(get_uid(taxmat[i,uniqueKnownTaxaLevel],verbose=FALSE,rows=1,ask=FALSE)[1])){
    ## Phylum
    out.phylum <- tax_name(query=taxmat[i,uniqueKnownTaxaLevel],get='Phylum',verbose=FALSE,ask=FALSE)$phylum
    # If null, second possibility to compute phylum, with rows=1:  First, it is better without rows=1, to avoid errors, for example, in bacilli, returning 'Nematoda' instead of 'Firmicutes'. If 'NA', it means several rows, and someone should be chosen, for example, the first one. The rows=1 is neccesary when the class level is really a phylum level, for example, with 'Actinobacteria', when several valid rows are returned.
    if(is.null(out.phylum)){
      out.phylum <- tax_name(query=taxmat[i,uniqueKnownTaxaLevel],get='Phylum',verbose=FALSE,ask=FALSE,rows=1)$phylum  
    }  
    if(!(is.null(out.phylum))){
      taxmat[i,"Phylum"] <- out.phylum
    }
    ## Kingdom
    out.kingdom <- tax_name(query=taxmat[i,uniqueKnownTaxaLevel],get='Kingdom',verbose=FALSE,ask=FALSE)$kingdom
    # If null, second possibility to compute phylum, with rows=1; as in Phylum.
    if(is.null(out.kingdom)){
      out.kingdom <- tax_name(query=taxmat[i,uniqueKnownTaxaLevel],get='Kingdom',verbose=FALSE,ask=FALSE,rows=1)$kingdom
    }
    if(!(is.null(out.kingdom))){
      taxmat[i,"Kingdom"] <- out.kingdom
    }
  } #if it exists UID
  # TO-IMPROVE: to generalize for all parent levels!!!
  # Also see classification()
  # Additional info: tax_rank(query="Mollicutes") --> rank: It returns the rank in the taxonomy
}
TAX = tax_table(taxmat)
data = phyloseq(OTU, TAX, MAP) 

