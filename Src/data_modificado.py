
import numpy as np
import pandas as pd
import random
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split


def read_data(random_state=42, #me da que esta funcion tambien habra que modificarla en algun momento porque en muchas funciones esta puesta esta y no la otra
              otu_filename='../../Datasets/otu_table_all_80.csv',
              metadata_filename='../../Datasets/metadata_table_all_80.csv'):
    otu = pd.read_csv(otu_filename, index_col=0, header=None).T
    otu = otu.set_index('otuids')
    otu = otu.astype('int32')
    metadata = pd.read_csv(metadata_filename)
    metadata = metadata.set_index('X.SampleID')
    domain = metadata[['KCAL',
                       'PROT',
                       'CARB']]
    #domain = pd.concat([domain, pd.get_dummies(domain['INBREDS'], prefix='INBREDS')], axis=1)
    #domain = pd.concat([domain, pd.get_dummies(domain['Maize_Line'], prefix='Maize_Line')], axis=1)
    #domain = domain.drop(['INBREDS', 'Maize_Line'], axis=1)
    df = pd.concat([otu, domain], axis=1, sort=True, join='outer')
    data_microbioma = df[otu.columns].to_numpy(dtype=np.float32)
    data_domain = df[domain.columns].to_numpy(dtype=np.float32)
    data_microbioma_train, data_microbioma_test, data_domain_train, data_domain_test = \
        train_test_split(data_microbioma, data_domain, test_size=0.1, random_state=random_state)
    return data_microbioma_train, data_microbioma_test, data_domain_train, data_domain_test, otu.columns, domain.columns



def read_df_with_transfer_learning_subset_fewerDomainFeatures(
              metadata_names=['age','Temperature','Precipitation3Days'],
              random_state=42,
              otu_filename='../Datasets/otu_table_all_80.csv',
              metadata_filename='../Datasets/metadata_table_all_80.csv'):
    otu = pd.read_csv(otu_filename, index_col=0, header=None).T
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
    df_microbioma_train, df_microbioma_no_train, df_domain_train, df_domain_no_train = \
        train_test_split(df_microbioma, df_domain, test_size=0.1, random_state=random_state)
    print("Dimensiones df_microbioma_train: "+str(df_microbioma_train.shape))
    print("Dimensiones df_microbioma_no_train: "+str(df_microbioma_no_train.shape))
    print("Dimensiones df_domain_train: "+str(df_domain_train.shape))
    print("Dimensiones df_domain_no_train: "+str(df_domain_no_train.shape))
    # Transfer learning subset
    df_microbioma_test, df_microbioma_transfer_learning, df_domain_test, df_domain_transfer_learning = train_test_split(df_microbioma_no_train, df_domain_no_train, test_size=0.1, random_state=random_state)
    df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test = train_test_split(df_microbioma_transfer_learning, df_domain_transfer_learning, test_size=0.3, random_state=random_state)
    
    return df_microbioma_train, df_microbioma_test, df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_train, df_domain_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test, otu.columns, domain.columns

def read_df_with_transfer_learning_subset(random_state=42,
              otu_filename='../Datasets/otu_table_all_80.csv',
              metadata_filename='../Datasets/metadata_table_all_80.csv'):
    otu = pd.read_csv(otu_filename, index_col=0, header=None, sep='\t').T
    otu = otu.set_index('otuids')
    otu = otu.astype('int32')
    metadata = pd.read_csv(metadata_filename, sep='\t')
    metadata = metadata.set_index('X.SampleID')
    domain = metadata[['age',
                       'Temperature',
                       'Precipitation3Days',
                       'INBREDS',
                       'Maize_Line']]
    domain = pd.concat([domain, pd.get_dummies(domain['INBREDS'], prefix='INBREDS')], axis=1)
    domain = pd.concat([domain, pd.get_dummies(domain['Maize_Line'], prefix='Maize_Line')], axis=1)
    domain = domain.drop(['INBREDS', 'Maize_Line'], axis=1)
    df = pd.concat([otu, domain], axis=1, sort=True, join='outer')
    #data_microbioma = df[otu.columns].to_numpy(dtype=np.float32)
    #data_domain = df[domain.columns].to_numpy(dtype=np.float32)
    df_microbioma = df[otu.columns]
    df_domain = df[domain.columns]    
    df_microbioma_train, df_microbioma_no_train, df_domain_train, df_domain_no_train = \
        train_test_split(df_microbioma, df_domain, test_size=0.1, random_state=random_state)
    df_microbioma_test, df_microbioma_transfer_learning, df_domain_test, df_domain_transfer_learning = \
        train_test_split(df_microbioma_no_train, df_domain_no_train, test_size=0.1, random_state=random_state)
    df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test = \
        train_test_split(df_microbioma_transfer_learning, df_domain_transfer_learning, test_size=0.3, random_state=random_state)
    
    return df_microbioma_train, df_microbioma_test, df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_train, df_domain_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test, otu.columns, domain.columns



def read_df_with_transfer_learning_subset_stratified_by_maize_line(random_state=42,
              otu_filename='../Datasets/otu_table_all_80.csv',
              metadata_filename='../Datasets/metadata_table_all_80.csv'):
    otu = pd.read_csv(otu_filename, index_col=0, header=None, sep='\t').T
    otu = otu.set_index('otuids')
    otu = otu.astype('int32')
    metadata = pd.read_csv(metadata_filename, sep='\t')
    metadata = metadata.set_index('X.SampleID')
    domain = metadata[['age',
                       'Temperature',
                       'Precipitation3Days',
                       'INBREDS',
                       'Maize_Line']]
    domain = pd.concat([domain, pd.get_dummies(domain['INBREDS'], prefix='INBREDS')], axis=1)
    domain = pd.concat([domain, pd.get_dummies(domain['Maize_Line'], prefix='Maize_Line')], axis=1)
    domain = domain.drop(['INBREDS', 'Maize_Line'], axis=1)
    df = pd.concat([otu, domain], axis=1, sort=True, join='outer')
    #data_microbioma = df[otu.columns].to_numpy(dtype=np.float32)
    #data_domain = df[domain.columns].to_numpy(dtype=np.float32)
    df_microbioma = df[otu.columns]
    df_domain = df[domain.columns]    
    df_microbioma_train, df_microbioma_no_train, df_domain_train, df_domain_no_train = \
        train_test_split(df_microbioma, df_domain, test_size=0.1, random_state=random_state)
    df_microbioma_test, df_microbioma_transfer_learning, df_domain_test, df_domain_transfer_learning = \
        train_test_split(df_microbioma_no_train, df_domain_no_train, test_size=0.1, random_state=random_state)
    df_temp=df_domain_transfer_learning
    col_stratify=df_temp.iloc[:,30:36][df==1].stack().reset_index().loc[:,'level_1']
    df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test = \
        train_test_split(df_microbioma_transfer_learning, df_domain_transfer_learning, test_size=0.3, random_state=random_state, stratify = col_stratify)
    
    return df_microbioma_train, df_microbioma_test, df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_train, df_domain_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test, otu.columns, domain.columns


def read_df_with_transfer_learning_2otufiles_fewerDomainFeatures(
              metadata_names=['age','Temperature','Precipitation3Days'],
              random_state=42,
              otu_filename='../Datasets/otu_table_all_80.csv',
              metadata_filename='../Datasets/metadata_table_all_80.csv',
              otu_transfer_filename='../Datasets/Walters5yearsLater/otu_table_Walters5yearsLater.csv',
              metadata_transfer_filename='../Datasets/Walters5yearsLater/metadata_table_Walters5yearsLater.csv'):
    otu = pd.read_csv(otu_filename, index_col=0, header=None, sep='\t').T
    otu = otu.set_index('otuids')
    otu = otu.astype('int32')
    metadata = pd.read_csv(metadata_filename, sep='\t')
    metadata = metadata.set_index('X.SampleID')
    domain = metadata[metadata_names]
    if 'INBREDS' in metadata_names:
        domain = pd.concat([domain, pd.get_dummies(domain['INBREDS'], prefix='INBREDS')], axis=1)
        domain = domain.drop(['INBREDS'], axis=1)
    elif 'Maize_Line' in metadata_names:
        domain = pd.concat([domain, pd.get_dummies(domain['Maize_Line'], prefix='Maize_Line')], axis=1)
        domain = domain.drop(['Maize_Line'], axis=1) 
    df = pd.concat([otu, domain], axis=1, sort=True, join='outer')
    df_microbioma = df[otu.columns]
    df_domain = df[domain.columns]    
    df_microbioma_train, df_microbioma_no_train, df_domain_train, df_domain_no_train = \
        train_test_split(df_microbioma, df_domain, test_size=0.1, random_state=random_state)
    df_microbioma_test, _, df_domain_test, _ = \
        train_test_split(df_microbioma_no_train, df_domain_no_train, test_size=0.1, random_state=random_state)
    otu_columns = otu.columns
    domain_columns = domain.columns
    # TRANSFER LEARNING SUBSETS
    otu = pd.read_csv(otu_transfer_filename, index_col=0, header=None, sep='\t').T
    otu = otu.set_index('otuids')
    otu = otu.astype('int32')
    metadata = pd.read_csv(metadata_transfer_filename, sep='\t')
    metadata = metadata.set_index('X.SampleID')
    domain = metadata[metadata_names]
    if 'INBREDS' in metadata_names:
        domain = pd.concat([domain, pd.get_dummies(domain['INBREDS'], prefix='INBREDS')], axis=1)
        domain = domain.drop(['INBREDS'], axis=1)
    elif 'Maize_Line' in metadata_names:
        domain = pd.concat([domain, pd.get_dummies(domain['Maize_Line'], prefix='Maize_Line')], axis=1)
        domain = domain.drop(['Maize_Line'], axis=1) 
    df = pd.concat([otu, domain], axis=1, sort=True, join='outer')
    df_microbioma = df[otu.columns]
    df_domain = df[domain.columns]        
    df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test = \
        train_test_split(df_microbioma, df_domain, test_size=0.3, random_state=random_state)
    
    return df_microbioma_train, df_microbioma_test, df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_train, df_domain_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test, otu_columns, domain_columns


def read_df_with_transfer_learning_2otufiles_differentDomainFeatures(
              metadata_names=['age','Temperature','Precipitation3Days'],
              random_state=42,
              otu_filename='../Datasets/otu_table_all_80.csv',
              metadata_filename='../Datasets/metadata_table_all_80.csv',
              metadata_names_transfer=['pH', 'Nmin', 'N', 'C', 'C.N', 'Corg', 'soil_type', 'clay_fration', 'water_holding_capacity'],
              otu_transfer_filename='../Datasets/Maarastawi2018/otu_table_Order_Maarastawi2018.csv',
              metadata_transfer_filename='../Datasets/Maarastawi2018/metadata_table_Maarastawi2018.csv'):
    otu = pd.read_csv(otu_filename, index_col=0, header=None, sep='\t').T
    otu = otu.set_index('otuids')
    otu = otu.astype('int32')
    metadata = pd.read_csv(metadata_filename, sep='\t')
    metadata = metadata.set_index('X.SampleID')
    domain = metadata[metadata_names]
    if 'INBREDS' in metadata_names:
        domain = pd.concat([domain, pd.get_dummies(domain['INBREDS'], prefix='INBREDS')], axis=1)
        domain = domain.drop(['INBREDS'], axis=1)
    elif 'Maize_Line' in metadata_names:
        domain = pd.concat([domain, pd.get_dummies(domain['Maize_Line'], prefix='Maize_Line')], axis=1)
        domain = domain.drop(['Maize_Line'], axis=1) 
    df = pd.concat([otu, domain], axis=1, sort=True, join='outer')
    df_microbioma = df[otu.columns]
    df_domain = df[domain.columns]    
    df_microbioma_train, df_microbioma_no_train, df_domain_train, df_domain_no_train = \
        train_test_split(df_microbioma, df_domain, test_size=0.1, random_state=random_state)
    df_microbioma_test, _, df_domain_test, _ = \
        train_test_split(df_microbioma_no_train, df_domain_no_train, test_size=0.1, random_state=random_state)
    otu_columns = otu.columns
    domain_columns = domain.columns
    # TRANSFER LEARNING SUBSETS
    otu = pd.read_csv(otu_transfer_filename, index_col=0, header=None, sep='\t').T
    #otu = otu.set_index('otuids')
    otu = otu.reset_index()
    otu = otu.drop(['otuids','index'],axis=1)
    otu = otu.astype('int32')
    metadata = pd.read_csv(metadata_transfer_filename, sep='\t')
    metadata = metadata.set_index('X.SampleID')
    domain = metadata[metadata_names_transfer]
    if 'soil_type' in metadata_names_transfer:
        domain = pd.concat([domain, pd.get_dummies(domain['soil_type'], prefix='soil_type')], axis=1)
        domain = domain.drop(['soil_type'], axis=1)
    domain = domain.reset_index()
    domain = domain.drop(['X.SampleID'], axis=1)
    df = pd.concat([otu, domain], axis=1, sort=True, join='outer')
    df = df.dropna(subset=metadata_names_transfer)
    df_microbioma = df[otu.columns]
    df_domain = df[domain.columns]        
    df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test = \
        train_test_split(df_microbioma, df_domain, test_size=0.3, random_state=random_state)
    
    return df_microbioma_train, df_microbioma_test, df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_train, df_domain_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test, otu_columns, domain_columns



def read_df_with_transfer_learning_subset_3domainFeatures(random_state=42,
              otu_filename='../Datasets/otu_table_all_80.csv',
              metadata_filename='../Datasets/metadata_table_all_80.csv'):
    otu = pd.read_csv(otu_filename, index_col=0, header=None, sep='\t').T
    otu = otu.set_index('otuids')
    otu = otu.astype('int32')
    metadata = pd.read_csv(metadata_filename, sep='\t')
    metadata = metadata.set_index('X.SampleID')
    domain = metadata[['age',
                       'Temperature',
                       'Precipitation3Days']]
    df = pd.concat([otu, domain], axis=1, sort=True, join='outer')
    #data_microbioma = df[otu.columns].to_numpy(dtype=np.float32)
    #data_domain = df[domain.columns].to_numpy(dtype=np.float32)
    df_microbioma = df[otu.columns]
    df_domain = df[domain.columns]    
    df_microbioma_train, df_microbioma_no_train, df_domain_train, df_domain_no_train = \
        train_test_split(df_microbioma, df_domain, test_size=0.1, random_state=random_state)
    df_microbioma_test, df_microbioma_transfer_learning, df_domain_test, df_domain_transfer_learning = \
        train_test_split(df_microbioma_no_train, df_domain_no_train, test_size=0.1, random_state=random_state)
    df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test = \
        train_test_split(df_microbioma_transfer_learning, df_domain_transfer_learning, test_size=0.3, random_state=random_state)
    
    return df_microbioma_train, df_microbioma_test, df_microbioma_transfer_learning_train, df_microbioma_transfer_learning_test, df_domain_train, df_domain_test, df_domain_transfer_learning_train, df_domain_transfer_learning_test, otu.columns, domain.columns



class DatasetSequence(keras.utils.Sequence):

    def __init__(self, data_microbioma, data_domain, idx, latent_space,
                 batch_size, shuffle, random_seed,
                 encoder_domain, encoder_bioma):
        self.idx = idx
        self.data_microbioma = data_microbioma
        self.data_domain = data_domain
        self.zeros = np.zeros((batch_size, latent_space), dtype=data_domain.dtype)
        self.batch_size = batch_size
        self.shuffle = shuffle
        random.seed(random_seed)
        self.encoder_domain = encoder_domain
        self.encoder_bioma = encoder_bioma
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / float(self.batch_size)))

    def __getitem__(self, idx):
        idx_init = idx * self.batch_size
        idx_end = (idx + 1) * self.batch_size
        m = self.data_microbioma[self.idx[idx_init:idx_end]]
        d = self.data_domain[self.idx[idx_init:idx_end]]

        if self.encoder_bioma is not None and self.encoder_domain is not None:
            x = (m, d)
            y = (m, m, self.zeros)
        elif self.encoder_bioma is not None:
            x = m
            y = m
        elif self.encoder_domain is not None:
            x = d
            y = m
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.idx)
