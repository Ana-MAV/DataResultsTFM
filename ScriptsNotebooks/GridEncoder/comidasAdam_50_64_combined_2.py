#Este es el script de prueba a ver si funciona con las comidas enteras


#importamos librerias
import sys
sys.path.append('Src/')
from data_modificado import * #hay funciones que estan cambiadas en este script para adaptralas a nuestro dataset
from train_2 import * #este hubo que modificar una linea tambien
from transfer_learning import * #hubo que modificart lo mismo que en train_2
from test_functions import *
from layers import *
from utils import *
from loss import *
from metric import *
from results import *
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

#Ponemos algunas funciones que huubo que cambiar
def read_df(
              metadata_names=['age','Temperature','Precipitation3Days'],
              random_state=42,
              otu_filename='../Datasets/otu_table_all_80.csv',
              metadata_filename='../Datasets/metadata_table_all_80.csv'):
    otu = pd.read_csv(otu_filename, index_col=0, header=None).T
    #print(otu.head())
    otu = otu.set_index('otuids')
    otu = otu.astype('int32')
    metadata = pd.read_csv(metadata_filename)
    #print(metadata.head())
    metadata = metadata.set_index('X.SampleID')
    metadata.head()
    domain = metadata[metadata_names]
    #if 'INBREDS' in metadata_names:
    #    domain = pd.concat([domain, pd.get_dummies(domain['INBREDS'], prefix='INBREDS')], axis=1)
    #    domain = domain.drop(['INBREDS'], axis=1)
    #elif 'Maize_Line' in metadata_names:
    #    domain = pd.concat([domain, pd.get_dummies(domain['Maize_Line'], prefix='Maize_Line')], axis=1)
    #    domain = domain.drop(['Maize_Line'], axis=1) 
    df = pd.concat([otu, domain], axis=1, sort=True, join='outer')
    #print(df.head())
    #data_microbioma = df[otu.columns].to_numpy(dtype=np.float32)
    #data_domain = df[domain.columns].to_numpy(dtype=np.float32)
    df_microbioma = df[otu.columns]
    df_domain = df[domain.columns]
    df_domain.head()
    df_microbioma_train, df_microbioma_no_train, df_domain_train, df_domain_no_train = train_test_split(df_microbioma, df_domain, test_size=0.1, random_state=random_state)
    #print("Dimensiones df_microbioma_train: "+str(df_microbioma_train.shape))
    #print("Dimensiones df_microbioma_no_train: "+str(df_microbioma_no_train.shape))
    #print("Dimensiones df_domain_train: "+str(df_domain_train.shape))
    #print("Dimensiones df_domain_no_train: "+str(df_domain_no_train.shape))
    # Transfer learning subset
    df_microbioma_test, df_microbioma_transfer_learning, df_domain_test, df_domain_transfer_learning = train_test_split(df_microbioma_no_train, df_domain_no_train, test_size=0.1, random_state=random_state)
    df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test = train_test_split(df_microbioma_transfer_learning, df_domain_transfer_learning, test_size=0.3, random_state=random_state)
    
    return df_microbioma_train, df_microbioma_test, df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_train, df_domain_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test, otu.columns, domain.columns
    #return df_microbioma_train, df_microbioma_test, df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_train, df_domain_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test, otu.columns, domain.columns


def train_kfold_mod(model_fn, m_train, d_train, z_train, m_test, d_test, z_test,
                batch_size, epochs, train_callbacks):
    all_models = model_fn()
    model, encoder_bioma, encoder_domain, decoder_bioma = all_models
    metrics_prefix = None
    if encoder_bioma is not None and encoder_domain is not None:
        x_train = (m_train, d_train)
        y_train = (m_train, m_train, z_train)
        x_test = (m_test, d_test)
        y_test = (m_test, m_test, z_test)
    elif encoder_bioma is not None:
        x_train = m_train
        y_train = m_train
        x_test = m_test
        y_test = m_test
        metrics_prefix = 'bioma'
    elif encoder_domain is not None:
        x_train = d_train
        y_train = m_train
        x_test = d_test
        y_test = m_test
        metrics_prefix = 'domain'

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000).batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    r = model.fit(train_dataset,
                  epochs=epochs,
                  validation_data=val_dataset,
                  callbacks=train_callbacks,
                  verbose=0)
    if metrics_prefix is not None:
        old_keys = r.history
        r.history = {}
        for k, v in old_keys.items():
            if k == 'loss' or k == 'val_loss':
                new_key = k
            elif k.startswith('val_'):
                new_key = 'val_{}_{}'.format(metrics_prefix, k[4:])
            else:
                new_key = '{}_{}'.format(metrics_prefix, k)
            r.history[new_key] = v
    del val_dataset
    del train_dataset
    del x_train
    del y_train
    del x_test
    del y_test
    return r, all_models

def train_2(model_fn,
          data_microbioma,
          data_domain,
          latent_space=10,
          folds=5,
          epochs=20,
          batch_size=128,
          learning_rate_scheduler=ExpDecayScheluder(),
          random_seed=347,
          verbose=0):
    data_zeros_latent = np.zeros((data_microbioma.shape[0], latent_space), dtype=data_microbioma.dtype)
    results = []
    models = []
    train_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=epochs + 1, restore_best_weights=True)]
    if verbose >= 0:
        train_callbacks += [TqdmCallback(verbose=verbose)]
    if learning_rate_scheduler is not None:
        train_callbacks += [learning_rate_scheduler.make()]

    if folds <= 1:
        m_train, m_test = data_microbioma, data_microbioma
        d_train, d_test = data_domain, data_domain
        z_train, z_test = data_zeros_latent, data_zeros_latent
        tf.compat.v1.random.set_random_seed(random_seed)
        r, m = train_kfold(model_fn, m_train, d_train, z_train, m_test, d_test, z_test,
                           batch_size, epochs, train_callbacks)
        results.append(r)
        models.append(m)

    else: #EL PROBLEMA ESTA AQUI, QUE HACE FALTA UN 
        kf = KFold(n_splits=folds, random_state=random_seed, shuffle=True)
        tf.compat.v1.random.set_random_seed(random_seed)

        for train_index, test_index in kf.split(data_microbioma):
            m_train, m_test = data_microbioma[train_index], data_microbioma[test_index]
            #print(m_train)
            #d_train, d_test = data_domain[train_index], data_domain[test_index]
            if data_domain is None:
                d_train, d_test = None, None
            else:
                d_train, d_test = data_domain[train_index], data_domain[test_index]
            #print(d_train)
            #Esto de hacer el if else ha funcionado, pero no se si hace lo que debe bien
            z_train, z_test = data_zeros_latent[train_index], data_zeros_latent[test_index]
            r, m = train_kfold_mod(model_fn, m_train, d_train, z_train, m_test, d_test, z_test,
                               batch_size, epochs, train_callbacks)
            results.append(r)
            models.append(m)
    return results, models

def perform_experiment_2_mod(cv_folds, epochs, batch_size, learning_rate, optimizer,
                       learning_rate_scheduler, input_transform, output_transform,
                       reconstruction_loss, latent_space, layers,
                       activation, activation_latent,
                       data_microbioma_train, data_domain_train,
                       show_results=True, device='/CPU:0'): #Show results cambiado de False  aTrue
    if input_transform is not None:
        input_transform = input_transform()
    #----------    
    if output_transform is not None:
        output_transform = output_transform()
    #----------      
    if reconstruction_loss.__class__.__name__ == 'MakeLoss':
        reconstruction_loss = reconstruction_loss.make()
    else:
        reconstruction_loss = reconstruction_loss()
    domain_layers = [l // 16 for l in layers] ####que es esto???? Esto es para las capas del domain
    #print(domain_layers)
    bioma_autoencoder = " -> ".join(["b"] +
                                    [str(l) for l in layers] +
                                    [str(latent_space)] +
                                    [str(l) for l in reversed(layers)] +
                                    ["b"])
    #---------- 
    #esto solo se utiliza para el texto, es irrelevante para nuestro error
    if data_domain_train is not None:
        domain_autoencoder = " -> ".join(["d"] +
                                     [str(l) for l in domain_layers] +
                                     [str(latent_space)] +
                                     [str(l) for l in reversed(layers)] +
                                     ["b"])
        
    else: 
        domain_autoencoder = " "
    #---------- 
    #donde se usa domain autoencoder?
    in_transform_name = input_transform.__class__.__name__ if input_transform else "none"
    out_transform_name = output_transform.__class__.__name__ if output_transform else "none"
    lr_scheduler_text = learning_rate_scheduler[
        1] if learning_rate_scheduler is not None else "none"
    lr_text = learning_rate if learning_rate_scheduler is not None else "constant = {}".format(
        learning_rate)
    learning_rate_scheduler = learning_rate_scheduler[
        0] if learning_rate_scheduler is not None else None
    optimizer = optimizer(learning_rate=learning_rate)
    #---------- 
    experiment_parameters = [
        ("Input transform", in_transform_name),
        ("Output transform", out_transform_name),
        ("Reconstruction Loss", reconstruction_loss.__class__.__name__),
        ("Latent Space", latent_space),
        ("Bioma Autoencoder", bioma_autoencoder),
        ("Domain Autoencoder", domain_autoencoder),
        ("Activation Encoder", activation),
        ("Activation Decoder", activation),
        ("Activation Latent", activation_latent),
        ("CV folds", cv_folds),
        ("Epochs", epochs),
        ("Batch Size", batch_size),
        ("Learning Rate Scheduler", lr_scheduler_text),
        ("Learning Rate", lr_text),
        ("Optimizer", optimizer.__class__.__name__),
    ]
    #----------  
    if show_results:
        md_text = ""
        md_text += "| Parameter             | Value         |\n"
        md_text += "|:----------------------|:--------------|\n"
        for n, v in experiment_parameters:
            md_text += "| {} | {} |\n".format(n, v)

        display(Markdown(md_text))
    #------------
    def create_model(print_data=False):
        bioma_shape=data_microbioma_train.shape[1]
        
        if data_domain_train is not None:
            domain_shape=data_domain_train.shape[1]
            #print("data_domain_train!=None")
        else:
            domain_shape=None
            #print("data_domain_train==None")
        models = autoencoder(bioma_shape=bioma_shape,
                             #bioma_shape=717,
                             domain_shape=domain_shape,
                             output_shape=bioma_shape,
                             #output_shape=717,
                             latent_space=latent_space,
                             bioma_layers=layers, #Esto es lo de [512,316]
                             domain_layers=domain_layers, #Esto son cada una de las layers divididas por 16
                             input_transform=input_transform,
                             output_transform=output_transform,
                             activation_function_encoder=activation,
                             activation_function_decoder=activation,
                             activation_function_latent=activation_latent)
        #Entiendo analizando lo demas que aqui NO esta el error
        #la funcion autoencoder esta en model.py (es la unica funcion en ese script)
        
        model, encoder_bioma, encoder_domain, decoder_bioma = models

        if print_data:
            plot_models(model, encoder_bioma, encoder_domain, decoder_bioma)
        compile_train(model,
                      encoder_bioma=encoder_bioma,
                      encoder_domain=encoder_domain,
                      reconstruction_error=reconstruction_loss,
                      encoded_comparison_error=losses.MeanAbsoluteError(),
                      metrics=get_experiment_metrics(input_transform, output_transform),
                      optimizer=optimizer)
        
        #print("He acabado create_model :)")
        return model, encoder_bioma, encoder_domain, decoder_bioma
    #-----------
    create_model(print_data=False)
    #-----------
    #Esta en esta seccion el problema, en train_2
    #print(data_domain_train)
    #print(latent_space)
    with tf.device(device):
        results, models = train_2(create_model,
                                data_microbioma_train,
                                data_domain_train,
                                latent_space=latent_space,
                                folds=cv_folds,
                                epochs=epochs,
                                batch_size=batch_size,
                                learning_rate_scheduler=learning_rate_scheduler,
                                verbose=-1)
    #----------
    validation_results = print_results(results, show_results=show_results)
    if show_results:
        display(Markdown("*************"))

    return experiment_parameters + validation_results, models, results


#cargamos datos
nombres_metadatos = ["F_TOTAL","F_CITMLB","F_OTHER","F_JUICE","V_TOTAL","V_DRKGR","V_REDOR_TOTAL","V_REDOR_TOMATO","V_REDOR_OTHER","V_STARCHY_TOTAL","V_STARCHY_POTATO","V_STARCHY_OTHER","V_OTHER",\
                     "V_LEGUMES","G_TOTAL","G_WHOLE","G_REFINED","PF_TOTAL","PF_MPS_TOTAL","PF_MEAT","PF_CUREDMEAT","PF_ORGAN","PF_POULT","PF_SEAFD_HI","PF_SEAFD_LOW","PF_EGGS","PF_SOY","PF_NUTSDS",\
                     "PF_LEGUMES","D_TOTAL","D_MILK","D_YOGURT","D_CHEESE","OILS","SOLID_FATS","ADD_SUGARS","A_DRINKS"]
df_microbioma_train, df_microbioma_test, _, _, \
df_domain_train, df_domain_test, _, _, otu_columns, domain_columns = read_df(metadata_names=nombres_metadatos,otu_filename='otu_table_especies_80.csv',metadata_filename='metadatos_comidas.csv')

data_microbioma_train = df_microbioma_train.to_numpy(dtype=np.float32)
data_microbioma_test = df_microbioma_test.to_numpy(dtype=np.float32)
data_domain_train = df_domain_train.to_numpy(dtype=np.float32)
data_domain_test = df_domain_test.to_numpy(dtype=np.float32)

#Preparamos las combinaciones pertinentes
import itertools as it

import pandas as pd
#Aclaracion de los nombres por posicion:
 # - SDGo: el optimizador
 # - [1,2,3]: 0.1,0.01 y 0.001 respectivamente --> learning rate
 # - [5,10,15]: latent spaces
 # - [2,3,4]: numero de capas
 # - [softmax, sigmoid, relu, tanh]: funcion de activacion
ficheros = ["Adac152relu_54.csv","Adac252relu_54.csv","Adac352relu_54.csv",\
            "Adac1102relu_54.csv","Adac2102relu_54.csv","Adac3102relu_54.csv",\
            "Adac1152relu_54.csv","Adac2152relu_54.csv","Adac3152relu_54.csv",\
            "Adac153relu_54.csv","Adac253relu_54.csv","Adac353relu_54.csv",\
            "Adac1103relu_54.csv","Adac2103relu_54.csv","Adac3103relu_54.csv",\
            "Adac1153relu_54.csv","Adac2153relu_54.csv","Adac3153relu_54.csv",\
            "Adac154relu_54.csv","Adac254relu_54.csv","Adac354relu_54.csv",\
            "Adac1104relu_54.csv","Adac2104relu_54.csv","Adac3104relu_54.csv",\
            "Adac1154relu_54.csv","Adac2154relu_54.csv","Adac3154relu_54.csv",\
            "Adac152tanh_54.csv","Adac252tanh_54.csv","Adac352tanh_54.csv",\
            "Adac1102tanh_54.csv","Adac2102tanh_54.csv","Adac3102tanh_54.csv",\
            "Adac1152tanh_54.csv","Adac2152tanh_54.csv","Adac3152tanh_54.csv",\
            "Adac153tanh_54.csv","Adac253tanh_54.csv","Adac353tanh_54.csv",\
            "Adac1103tanh_54.csv","Adac2103tanh_54.csv","Adac3103tanh_54.csv",\
            "Adac1153tanh_54.csv","Adac2153tanh_54.csv","Adac3153tanh_54.csv",\
            "Adac154tanh_54.csv","Adac254tanh_54.csv","Adac354tanh_54.csv",\
            "Adac1104tanh_54.csv","Adac2104tanh_54.csv","Adac3104tanh_54.csv",\
            "Adac1154tanh_54.csv","Adac2154tanh_54.csv","Adac3154tanh_54.csv"]



for fichero in ficheros:
    try:
      open(fichero,"r")
      continue
    except:
        if int(fichero[4]) == 1:
            learning_rate_val = 0.1
        elif int(fichero[4]) == 2:
            learning_rate_val = 0.01
        else:
            learning_rate_val = 0.001
        
        if int(fichero[5]) == 1:
            latent_space_val=int(fichero[5:7])
            optimizer_val = fichero[8:-7]
            if int(fichero[7]) == 2:
                layers_val=[512,256]
            elif int(fichero[7]) == 3:
                layers_val=[512,256,128]
            else:
                layers_val=[512,256,128,64]
        else:
            latent_space_val=5
            optimizer_val=fichero[7:-7]
            if int(fichero[6]) == 2:
                layers_val=[512,256]
            elif int(fichero[6]) == 3:
                layers_val=[512,256,128]
            else:
                layers_val=[512,256,128,64]
        
        param_grid_2 = {'epochs': [50], 
              'batch_size': [64], 
              'learning_rate':[learning_rate_val],
              'optimizer':[optimizers.Adam],
              'latent_space':[latent_space_val],
              'layers':[layers_val],
              'activation':[optimizer_val],
              'activation_latent': ['softmax','sigmoid','relu','tanh']}
        combinations_2 = list(it.product(*(param_grid_2[Name] for Name in list(param_grid_2.keys()))))
      
        results_grid_combined = pd.DataFrame(columns = ["parameters","val_BrayCurtis"])
        for g in combinations_2:
        #entrenamos el autoencoder
            experiment_metrics, models, results = perform_experiment_2_mod(cv_folds=5,
                                                                       epochs=g[0], ##este es el que varia
                                                                       batch_size=g[1],
                                                                       learning_rate=g[2], ##este es el que varia
                                                                       optimizer=g[3],
                                                                       learning_rate_scheduler=None,
                                                                       input_transform=Percentage, #--> lo quitamos porque ya lo tenemos en relativo, con el 80 no
                                                                       output_transform=tf.keras.layers.Softmax,
                                                                       reconstruction_loss=MakeLoss(LossBrayCurtis, Percentage, None),
                                                                       latent_space=g[4],
                                                                       layers=g[5],
                                                                       activation=g[6],
                                                                       activation_latent=g[7],
                                                                       data_microbioma_train=data_microbioma_train,
                                                                       data_domain_train=data_domain_train,
                                                                       show_results=False,#esto se ha cambiado para que no me muestre los resultados
                                                                       device='/CPU:0')
            results_grid_combined = results_grid_combined.append({"parameters":g,"val_BrayCurtis":experiment_metrics[23][1]},ignore_index=True)
            print(str(g)+"------------------------------")
        ####Esto habra que volverlo a reescribir porque algo de las tabulaciones no va bien, porque no se escribe el ultimo de los tanhs
        results_grid_combined.to_csv(fichero,index=False)
    break