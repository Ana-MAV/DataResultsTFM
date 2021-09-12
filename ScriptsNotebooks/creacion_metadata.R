#script para poder poner bien los metadatos
#por ahora vamos a hacerlo con los nutrientes y kcal, que es lo mas facil de hacer metadatos


setwd("C:/Users/anama/OneDrive/Documentos/2020-2021/practicas/transcurso/dietstudy_analyses-master")

original_table <- read.table("data/diet/raw_and_preprocessed_ASA24_data/Totals_to_use.txt",header = TRUE,sep="\t")
table_microbiome_especies <- read.csv("resultados_ana/otus_todosordenes.csv") #######--------

#en original_table (metadatos) hay mas muestras de las que tenemos de analisis de otus
#hay que filtrar --> obtenemos los metadatos de los que tenemos muestras de otus

nombres_columnas <- names(table_microbiome_especies)[2:527]

metadata_filtrada <- original_table[original_table$X.SampleID %in% nombres_columnas,]

#-----------------------

#hay muestras de las que tenemos otus pero no metadata, por lo que hay que quitarlos tambien
samples_sin_metadatos <- c()
for (columna in names(table_microbiome_especies)){
  if (!(columna %in% original_table$X.SampleID)){
    print(columna)
  }
}
#Hay que quitar 0077,0078,0080,0081 y 0083
table_microbiome_especies$MCT.f.0077 <- NULL
table_microbiome_especies$MCT.f.0078 <- NULL
table_microbiome_especies$MCT.f.0080 <- NULL
table_microbiome_especies$MCT.f.0081 <- NULL
table_microbiome_especies$MCT.f.0083 <- NULL


#Ahora tenemos los datos de otus con todas las samples de metadatos, lo guardamos
####esto habria que hacerlos con todos si queremos trabajr con cosas que no sean especies
write.csv(table_microbiome_especies,"resultados_ana/otus_todos_conMetadatos.csv",row.names=FALSE)#-------

#ahora vamos a filtrar las columnas que no queremos de los metadatos
names(metadata_filtrada)
#queremos quitar Replicate, UserName, StudyDayNo y RecordDayNo
metadata_filtrada$Replicate <- NULL
metadata_filtrada$UserName <- NULL
metadata_filtrada$StudyDayNo <- NULL
metadata_filtrada$RecordDayNo <- NULL

#guardamos el archvo de metadata filtrado
write.csv(metadata_filtrada,"resultados_ana/todos_metadatos_todos.csv",row.names=FALSE)#----------
