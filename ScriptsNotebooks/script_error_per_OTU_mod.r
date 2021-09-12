###Este es el mio, el original tiene el mismo nombre pero sin el mod y en la carpeta Src
setwd("2020-2021/practicas/transcurso/dietstudy_analyses-master")

library(ggplot2)
library(RColorBrewer)
myPalette <- brewer.pal(12, "Set3")

### Function name_from_OTU_id ###
name_from_OTU_id <- function(dic, id, tax_level){
  return(unlist(as.character(dic[id,tax_level])))
}

#Para las comidas, hay que cambiarlo para la filtrada y no filtrada
for (tipo in c("OTU","Combined")){
  file_error_perOTU = paste("Results/errors_perOTU_",tipo,"_comida_todos.tsv",sep="")#----------------
  file_tax = 'resultados_ana/nuevos_datos/tax_table_con_aggrecate.csv'
  taxmat <- read.table(file_tax,sep=',',row.names=1,header=TRUE)
  #quitamos columnas que son para nuetsra info 
  colnames(taxmat) = c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species","Subspecie")
  # Read table with one row per OTU, and different columns with different errors (RSD, RSE, RRSE)
  error_perOTU <- read.table(file_error_perOTU,sep='\t',row.names=1,header=TRUE,check.names=FALSE)
  # Add taxonomic info to each OTU --> esto se ha modificado porque teniamos de diferenets maneras el diccionario
  for(level in colnames(taxmat)){
    print(level)
    for (id in row.names(error_perOTU)){
      error_perOTU[id,level] = name_from_OTU_id(taxmat,id,level)
    }
  }
  # Sort df to obtain best predicted OTUs
  error_perOTU_sorted <- error_perOTU[order(error_perOTU$RRSE),]
  write.table(error_perOTU_sorted, paste('Results/comida/errors_perOTU_',tipo,'comida_withTaxaNames.tsv',sep=""), sep='\t', row.names = TRUE, col.names = NA)#-------------
}

#Para los nutrientes
for (tipo in c("OTU","Combined")){
  for (variables in c("22","44")){
    file_error_perOTU = paste("Results/errors_perOTU_",variables,tipo,"_nutrientes.tsv",sep="")#----------------
    file_tax = 'resultados_ana/nuevos_datos/tax_table_con_aggrecate.csv'
    taxmat <- read.table(file_tax,sep=',',row.names=1,header=TRUE)
    #quitamos columnas que son para nuetsra info 
    colnames(taxmat) = c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species","Subspecie")
    # Read table with one row per OTU, and different columns with different errors (RSD, RSE, RRSE)
    error_perOTU <- read.table(file_error_perOTU,sep='\t',row.names=1,header=TRUE,check.names=FALSE)
    # Add taxonomic info to each OTU --> esto se ha modificado porque teniamos de diferenets maneras el diccionario
    for(level in colnames(taxmat)){
      print(level)
      for (id in row.names(error_perOTU)){
        error_perOTU[id,level] = name_from_OTU_id(taxmat,id,level)
      }
    }
    # Sort df to obtain best predicted OTUs
    error_perOTU_sorted <- error_perOTU[order(error_perOTU$RRSE),]
    write.table(error_perOTU_sorted, paste('Results/errors_perOTU_',variables,tipo,'nutrientes_withTaxaNames.tsv',sep=""), sep='\t', row.names = TRUE, col.names = NA)#-------------
  }
}



#--------------------------------------------
##Esta parte tiene que personalizarse de otra manera porque son imagenes y demas historias
# Compute % of each Phylum in all 599 OTUs
#------------------------
#Esto es para comidas
for (tipo in c("OTU","Combined")){
  print(tipo)
  error_perOTU_figuras = read.table(paste('Results/errors_perOTU_',tipo,'comida_withTaxaNames.tsv',sep=""), sep='\t',header = TRUE,row.names=1)
  error_perOTU_sorted_by_RRSE <- error_perOTU_figuras[order(error_perOTU_figuras$RRSE),]
  for (nivel_1 in c("Phylum", "Class", "Order", "Family", "Genus")){
    
    numOTUs=nrow(error_perOTU_figuras)
    df_perc_OTU = data.frame(perc=double())
    
    for(phy in sort(unique(error_perOTU_figuras[,nivel_1]))){
      
      perc=nrow(subset(error_perOTU_figuras[error_perOTU_figuras[nivel_1]==phy,]))/numOTUs
      df_perc_OTU[phy,'perc']=perc
      
      }
    #print(df_perc_OTU) --> se printea si lo queremos tener o ver, quizas estaria bien guardarlo
    #pie(df_perc_OTU$perc,labels=row.names(df_perc_OTU),col=myPalette,main=paste('OTUs distribution (by ',nivel_1,')',sep=""))
    #Este pie es igual para todos, porque es la distribucion de los OTUs a lo largo de los datos --> se guarda 1 vez para todos
    #----------------
    #Esta ya es la parte unica
    # Compute 5% best predicted OTUs
    #599*0.05=29.95 ~ 30
    best_predicted_OTUs=head(error_perOTU_sorted_by_RRSE[,c('RRSE',nivel_1)],30)
    best_predicted_OTUs[order(best_predicted_OTUs[,nivel_1]),]
    
    # Percentage of phylums with OTUs with the best 5% stability
    numOTUs=30
    df_best_predicted = data.frame(perc=double())
    for(phy in sort(unique(best_predicted_OTUs[,nivel_1]))){
      perc=nrow(subset(best_predicted_OTUs[best_predicted_OTUs[nivel_1]==phy,]))/numOTUs
      df_best_predicted[phy,'perc']=perc
      #print(paste(phy,perc))
    }
    #df_best_predicted
    pie(df_best_predicted$perc,labels=row.names(df_best_predicted),col=myPalette,main=paste('5% best predicted OTUs (by ',nivel_1,')',sep=""))
    }
}

#Esto es para nutrientes
################falta por hacer, pero es copiar y pegar y cambiar un poco algunas cosas
for (tipo in c("OTU","Combined")){
  print(tipo)
  for (variables in c("22","44")){
    print(variables)
    error_perOTU_figuras = read.table(paste('Results/errors_perOTU_',variables,tipo,'nutrientes_withTaxaNames.tsv',sep=""), sep='\t',header = TRUE,row.names=1)
    error_perOTU_sorted_by_RRSE <- error_perOTU_figuras[order(error_perOTU_figuras$RRSE),]
    for (nivel_1 in c("Phylum", "Class", "Order", "Family", "Genus")){
      
      numOTUs=nrow(error_perOTU_figuras)
      df_perc_OTU = data.frame(perc=double())
      
      for(phy in sort(unique(error_perOTU_figuras[,nivel_1]))){
        
        perc=nrow(subset(error_perOTU_figuras[error_perOTU_figuras[nivel_1]==phy,]))/numOTUs
        df_perc_OTU[phy,'perc']=perc
        
      }
      #print(df_perc_OTU) --> se printea si lo queremos tener o ver, quizas estaria bien guardarlo
      #pie(df_perc_OTU$perc,labels=row.names(df_perc_OTU),col=myPalette,main=paste('OTUs distribution (by ',nivel_1,')',sep=""))
      #Este pie es igual para todos, porque es la distribucion de los OTUs a lo largo de los datos --> se guarda 1 vez para todos
      #----------------
      #Esta ya es la parte unica
      # Compute 5% best predicted OTUs
      #599*0.05=29.95 ~ 30
      best_predicted_OTUs=head(error_perOTU_sorted_by_RRSE[,c('RRSE',nivel_1)],30)
      best_predicted_OTUs[order(best_predicted_OTUs[,nivel_1]),]
      
      # Percentage of phylums with OTUs with the best 5% stability
      numOTUs=30
      df_best_predicted = data.frame(perc=double())
      for(phy in sort(unique(best_predicted_OTUs[,nivel_1]))){
        perc=nrow(subset(best_predicted_OTUs[best_predicted_OTUs[nivel_1]==phy,]))/numOTUs
        df_best_predicted[phy,'perc']=perc
        #print(paste(phy,perc))
      }
      #df_best_predicted
      pie(df_best_predicted$perc,labels=row.names(df_best_predicted),col=myPalette,main=paste('5% best predicted OTUs (by ',nivel_1,')',sep=""))
    }
  }
}

#-------------------

# #pdf('../Results/nutrientes/piechart_perc_all_OTUs.pdf')
# pie(df_perc_OTU$perc,labels=row.names(df_perc_OTU),col=myPalette,main='OTUs distribution (by Phylum)')
# #dev.off()
# 
# 
# # Compute 5% best predicted OTUs
# error_perOTU_sorted_by_RRSE <- error_perOTU[order(error_perOTU$RRSE),]
# #405*0.05=20.25
# best_predicted_OTUs=head(error_perOTU_sorted_by_RRSE[,c('RRSE','Genus')],20)
# best_predicted_OTUs[order(best_predicted_OTUs$Genus),]
# 
# # Percentage of phylums with OTUs with the best 5% stability
# numOTUs=20
# df_best_predicted = data.frame(perc=double())
# for(phy in sort(unique(best_predicted_OTUs$Genus))){
#   perc=nrow(subset(best_predicted_OTUs[best_predicted_OTUs$Genus==phy,]))/numOTUs
#   df_best_predicted[phy,'perc']=perc
#   #print(paste(phy,perc))
# }
# df_best_predicted
#   
# 
# 
# #pdf('../Results/piechart_5perc_best_predicted_OTUs.pdf')
# pie(df_best_predicted$perc,labels=row.names(df_best_predicted),col=myPalette,main='5% best predicted OTUs (by Genus)')
# #dev.off()


