setwd("C:/Users/anama/OneDrive/Documentos/2020-2021/practicas/transcurso/dietstudy_analyses-master")

original_table <- read.table("data/diet/raw_and_preprocessed_ASA24_data/Totals_to_use.txt",header = TRUE,sep="\t")
table_microbiome_especies <- read.csv("resultados_ana/otus_todosordenes.csv") #######--------

#We have more metadata that microbial samples, we need to remove them
#We have microbial data that does not have metadata, we need to remove them also

nombres_columnas <- names(table_microbiome_especies)[2:527]

metadata_filtrada <- original_table[original_table$X.SampleID %in% nombres_columnas,]

#-----------------------

samples_sin_metadatos <- c()
for (columna in names(table_microbiome_especies)){
  if (!(columna %in% original_table$X.SampleID)){
    print(columna)
  }
}
#Remove 0077,0078,0080,0081 and 0083 samples
table_microbiome_especies$MCT.f.0077 <- NULL
table_microbiome_especies$MCT.f.0078 <- NULL
table_microbiome_especies$MCT.f.0080 <- NULL
table_microbiome_especies$MCT.f.0081 <- NULL
table_microbiome_especies$MCT.f.0083 <- NULL



write.csv(table_microbiome_especies,"resultados_ana/otus_todos_conMetadatos.csv",row.names=FALSE)#-------

#We filter columns that are extra information, not nutrtinional or food data
names(metadata_filtrada)
metadata_filtrada$Replicate <- NULL
metadata_filtrada$UserName <- NULL
metadata_filtrada$StudyDayNo <- NULL
metadata_filtrada$RecordDayNo <- NULL

#save data
write.csv(metadata_filtrada,"resultados_ana/todos_metadatos_todos.csv",row.names=FALSE)#----------
