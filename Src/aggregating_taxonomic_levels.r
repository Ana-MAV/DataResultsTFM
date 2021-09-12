library(phyloseq)
setwd("C:/Users/anama/OneDrive/Documentos/2020-2021/practicas/transcurso/dietstudy_analyses-master/resultados_ana")

otumat <- read.table('otu_table_todos_80.csv',sep=',',row.names=1,header=TRUE,check.names=FALSE)
OTU = otu_table(otumat, taxa_are_rows = TRUE)

mapmat <- read.table('todos_metadatos_todos.csv',sep=',',row.names=1,header=TRUE)
MAP = sample_data(mapmat)


taxmat <- read.table('tax_table_todos_80.csv',sep=',',row.names=1,header=TRUE)
colnames(taxmat) = c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species","Subspecie")
taxmat=as.matrix(taxmat)
TAX = tax_table(taxmat)

tSpecies = phyloseq(OTU, MAP, TAX) 
save(tSpecies,file='borrar/table_all_80_phyloseqObject.RData')

#El problema de esta fncion es que se va reseteando el indice y no quedan bien los otuids
#No se puede formar un phyloseq con otra columna de los otuids, por lo que hay que hacer una funcion nosotros
for (taxLevel in colnames(tax_table(tSpecies))[2:7]){
  print(taxLevel)
  tAgg=tax_glom(tSpecies,taxrank=taxLevel)
  # Save phyloseq object
  save(tAgg,file=paste('borrar/table',taxLevel,'phyloseqObject.RData',sep='_'))
  # Save .tsv
  df.otu=as.data.frame(otu_table(tAgg))
  otuids=rownames(df.otu)
  print(otuids)
  data=cbind(otuids, df.otu)
  write.table(data, paste('borrar/otu_table_',taxLevel,'.csv',sep=''), sep=",", row.names = FALSE, col.names = TRUE)
  df.tax=as.matrix(tax_table(tAgg))
  #print('after')
  otuids=row.names(tax_table(tAgg))
  df.tax=cbind(otuids, df.tax)
  write.table(df.tax, paste('borrar/tax_table_',taxLevel,'.csv',sep=''), sep=",", row.names = FALSE, col.names = TRUE)
  #df.sample=as.data.frame(sample_data(tAgg))
  #write.table(df.sample, '../Datasets/Aggregated/metadata_table_all_80.csv', sep="\t", row.names = FALSE, col.names = TRUE)
}
