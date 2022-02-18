import argparse
import logging
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from clearml import Task
from sklearn.model_selection import train_test_split
from clearml import StorageManager

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def segregate_dataset(): 
    manager = StorageManager()
    
    dataset_path = manager.get_local_copy(
        remote_url="https://files.clear.ml/a%20ML%20example/Prepreocessing.71845909e9b643fca92e5902c32265a1/artifacts/processed_data/preprocessed_data.csv"
    )

    df = pd.read_table(dataset_path,sep=',',encoding='utf-8')

    task = Task.init(project_name="a ML example",task_name="segregate data")
    df['descricao'].replace({'APROVADO':'1',"FALHOU":"0","REPROVADO POR NOTA E FALTA":"0"},inplace=True)
    x_train , x_test , y_train , y_test = train_test_split(df.drop(columns={'descricao'}),
                                                        df.descricao,
                                                        stratify=df.descricao,
                                                        test_size=0.3)
    df_train = x_train
    df_train['descricao'] = y_train

    df_test = x_test
    df_test['descricao'] = y_test

    filename_train = "segregation/train_data.csv"

    df_train.to_csv(filename_train,index=False)

    task.upload_artifact(name="train_data", artifact_object=filename_train,metadata=dict(
                                    description="Dados para treino do modelo"
    ))

    filename_test = "segregation/test_data.csv"

    df_test.to_csv(filename_test,index=False)
    # Create a new artifact and upload

    task.upload_artifact(name="test_data", artifact_object=filename_test,metadata=dict(
                                    description="Dados para teste do modelo"
    ))