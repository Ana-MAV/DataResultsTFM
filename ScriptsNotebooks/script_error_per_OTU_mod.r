###Original in Src folder, this is customized to my data
setwd("2020-2021/practicas/transcurso/dietstudy_analyses-master")

library(ggplot2)
library(RColorBrewer)
myPalette <- brewer.pal(12, "Set3")

name_from_OTU_id <- function(dic, id, tax_level){
  return(unlist(as.character(dic[id,tax_level])))
}

#Food info
for (tipo in c("OTU","Combined")){
  file_error_perOTU = paste("Results/errors_perOTU_",tipo,"_comida_todos.tsv",sep="")#----------------
  file_tax = 'resultados_ana/nuevos_datos/tax_table_con_aggrecate.csv'
  taxmat <- read.table(file_tax,sep=',',row.names=1,header=TRUE)
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

#Nutrient info
for (tipo in c("OTU","Combined")){
  for (variables in c("22","44")){
    file_error_perOTU = paste("Results/errors_perOTU_",variables,tipo,"_nutrientes.tsv",sep="")#----------------
    file_tax = 'resultados_ana/nuevos_datos/tax_table_con_aggrecate.csv'
    taxmat <- read.table(file_tax,sep=',',row.names=1,header=TRUE)
    colnames(taxmat) = c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species","Subspecie")
    # Read table with one row per OTU, and different columns with different errors (RSD, RSE, RRSE)
    error_perOTU <- read.table(file_error_perOTU,sep='\t',row.names=1,header=TRUE,check.names=FALSE)
    # Add taxonomic info to each OTU
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
#Food
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

#Nutrient
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