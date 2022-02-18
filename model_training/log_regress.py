"""
Creators: Diego Medeiros e Reyne Jasson
create a pipeline for building a logistic regression model 
and study how does the corona virus changed the sucess 
on school.                                                                                                                     
"""



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline, FeatureUnion


from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

from joblib import dump

import argparse

from clearml import Task


#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self.feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self.feature_names ]


class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    # Class constructor method that takes one boolean as its argument
    def __init__(self, new_features=True, colnames=None):
        self.new_features = new_features
        self.colnames = colnames

    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 

    def get_feature_names(self):
        return self.colnames.tolist()

    # Transformer method we wrote for this transformer 
    def transform(self, X , y = None):
        df = pd.DataFrame(X,columns=self.colnames)
        
        columns = self.colnames
        # Create new features with label Encoding

        df['grau_academico'].replace({'BACHARELADO':'3', 'LICENCIATURA':'2', 
                                      'TECNOLÓGICO':'1',"OUTRO":"0"},inplace=True)


        print(df.head())
        # update column names  
        return df

class NumericalTransformer( BaseEstimator, TransformerMixin ):
    # Class constructor method that takes a model parameter as its argument
    # model 0: minmax
    # model 1: standard
    # model 2: without scaler
    def __init__(self, model = 0, colnames=None):
        self.model = model
        self.colnames = colnames

    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self

    # return columns names after transformation
    def get_feature_names(self):
        return self.colnames 

    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):
        df = pd.DataFrame(X,columns=self.colnames)

        for coluna in self.colnames:
          df[coluna] = pd.to_numeric(df[coluna],errors='coerce')
        # update columns name
        df.fillna(value=0,inplace=True)
        self.colnames = df.columns.tolist()

        df['idade'] = 2020 - df['ano_nascimento'].astype(int)

        # minmax
        if self.model == 0: 
            scaler = MinMaxScaler()
            # transform data
            df = scaler.fit_transform(df)

        elif self.model == 1:
            scaler = StandardScaler()
            # transform data
            df = scaler.fit_transform(df)
        else:
            df = df.values

        return df

def process_args(ARGS:dict,task:Task):

    logger = task.get_logger()

    preprocessed_data_task = Task.get_task(task_id=ARGS.task_id)
    # access artifact
    local_csv = preprocessed_data_task.artifacts[ARGS.dataset_name].get_local_copy()
    

    data = pd.read_csv(local_csv,encoding='utf-8',sep=',',dtype=object)


    data['ano_nascimento'].fillna(value='2000',inplace=True)
    #create age feature
    
    data['ano_nascimento'] = data['ano_nascimento'].astype(int)
    data['renda'] = data['renda'].astype(int)
    # Spliting train.csv into train and validation dataset
    print("Spliting data into train/val")

    #label replacement
    # Create logical instance from multivalue_feture
    data['local_ou_de_fora'] = (data['estado_origem']==('Rio Grande do Norte'))

    # Fill nan  for "Outro" category
    data['raca'].fillna(value='Não Informado',inplace=True)
    data['area_conhecimento'].fillna(value='Outra',inplace=True)
    data['grau_academico'].fillna(value='OUTRO',inplace=True)

    # Start label Encoder
    

    data.drop(columns={'estado_origem','cidade_origem'},inplace=True)       

    data['descricao'].replace({'APROVADO':'1',"FALHOU":"0","REPROVADO POR NOTA E FALTA":"0"},inplace=True)
    data['descricao'] = pd.to_numeric(data['descricao'],errors='coerce')
    data['descricao'].fillna(value='0',inplace=True)
    
    print(data.dtypes)
    # split-out train/validation and test dataset
    x_train,x_val,y_train,y_val = train_test_split(data.drop(columns=['descricao']),
                                                      data['descricao'],
                                                      test_size=0.2,
                                                      random_state=2,
                                                      shuffle=True,
                                                    stratify = data['descricao'] if ARGS.stratify else None)
    
    print("x train: {}".format(x_train.shape))
    print("y train: {}".format(y_train.shape))
    print("x val: {}".format(x_val.shape))
    print("y val: {}".format(y_val.shape))
    print("x train: {}".format(list(x_train.columns)))
    print("Removal Outliers")
    # temporary variable
    x = x_train.select_dtypes("int64").copy()

    # identify outlier in the dataset
    lof = LocalOutlierFactor()
    outlier = lof.fit_predict(x)
    mask = (outlier != -1)

    print("x_train shape [original]: {}".format(x_train.shape))
    print("x_train shape [outlier removal]: {}".format(x_train.loc[mask,:].shape))

    # dataset without outlier, note this step could be done during the preprocesing stage
    x_train = x_train.loc[mask,:].copy()
    y_train = y_train[mask].copy()
    print("Encoding Target Variable")
    # define a categorical encoding for target variable
    le = LabelEncoder()

    # fit and transform y_train
    y_train = le.fit_transform(y_train)
    # transform y_test (avoiding data leakage)
    y_val = le.transform(y_val)
    print(y_train)
    print("Classes [0, 1]: {}".format(le.inverse_transform([0, 1])))
    
    # Pipeline generation
    print("Pipeline generation")
    
    # Categrical features to pass down the categorical pipeline 
    categorical_features = x_train.select_dtypes(["object",'bool']).columns.to_list()

    # Numerical features to pass down the numerical pipeline 
    numerical_features = x_train.select_dtypes("int64").columns.to_list()
    # Defining the steps in the categorical pipeline 

    categorical_pipeline = Pipeline(steps = [('cat_selector',FeatureSelector(categorical_features)),
                                            ('imputer_cat', SimpleImputer(strategy="most_frequent")),
                                             ('cat_encoder',OneHotEncoder(sparse=False,drop="first"))
                                            ]
                                   )
    # Defining the steps in the numerical pipeline     
    print(FeatureSelector(numerical_features))
    
    numerical_pipeline = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),
                                           ('imputer_cat', SimpleImputer(strategy="median")),
                                           ('num_transformer', NumericalTransformer(colnames=numerical_features))
                                          ]
                                 )

    # Combining numerical and categorical piepline into one full big pipeline horizontally 
    # using FeatureUnion
    full_pipeline_preprocessing = FeatureUnion(transformer_list = [('cat_pipeline', categorical_pipeline),
                                                                   ('num_pipeline', numerical_pipeline)
                                                                  ]
                                              )
   
    # The full pipeline 
    pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),
                             ("classifier",LogisticRegression())
                            ]
                   )

    # training 
    print("Training{}".format(list(x_train.dtypes)))
    pipe.fit(x_train,y_train)

    # predict
    print("Infering")
    predict = pipe.predict(x_val)

    print(predict)

    return pipe,x_val,y_val



if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="The training script",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to clearML",
        default='regressao_logistica.joblib'
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='processed_data',
        help="The dataset name to generate model"

    )

    parser.add_argument(
        "--task_id",
        type=str,
        help="Task ID where the data was generated",
        default='71845909e9b643fca92e5902c32265a1'
    )


    parser.add_argument(
        "--stratify",
        type=int,
        help="Name for column which to stratify",
        default=None
    )

    ARGS = parser.parse_args()

    task = Task.init(project_name="a ML example",task_name="logist training")


    # process the arguments

    clf,x_val,y_val = process_args(ARGS,task)

    y_predict = clf.predict(x_val)

    #ClearML will automatically save anything reported to matplotlib!
    cm = confusion_matrix(y_true=y_val,y_pred=y_predict,normalize='true')
    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    sns.heatmap(cm,cmap=cmap, annot=True)
    plt.show()

    print(f"Exporting model {ARGS.model_export}")
    dump(clf, ARGS.model_export)
    task.upload_artifact("log_regress_classifier", ARGS.model_export)

