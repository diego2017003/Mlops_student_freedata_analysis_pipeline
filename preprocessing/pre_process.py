"""
Creator: Reyne
Date: 11 feb. 2022
After download the raw data we need to preprocessing it.
At the end of this stage we have been created a new artfiact (clean_data).
"""

import argparse
import logging
import os
import pandas as pd

from clearml import Task

def process_args(args,task):
    """
    Arguments
        args - command line arguments
        args.input_artifact: Fully qualified name for the raw data artifact
        args.artifact_name: Name for the artifact that will be created
        args.artifact_description: Description for the artifact
    """
    
    # create a new wandb project
    logger = task.get_logger()

    raw_data_task = Task.get_task(task_id=args.task_id)
    # access artifact
    local_csv = raw_data_task.artifacts[args.input_artifact].get_local_copy()
    
    
    # columns used 
    columns_keep = ['sexo', 'ano_nascimento', 'raca', 'estado_origem',
                    'cidade_origem','renda','area_conhecimento','possui_auxilio_alimentacao', 
                    'possui_auxilio_transporte','possui_auxilio_residencia_moradia',
                    'grau_academico','descricao']
    
    # create a dataframe from the artifact path
    df = pd.read_csv(local_csv,usecols=columns_keep)


    mapping ={
     "APROVADO POR NOTA":"APROVADO",
     "EXCLUIDA":"FALHOU",
    "CANCELADO":"FALHOU",
    "REPROVADO POR MÉDIA E POR FALTAS":"FALHOU",
    "REPROVADO":"FALHOU",
    "REPROVADO POR NOTA E FALTA": "FALHOU"
    }

    df = df.query('descricao != "INDEFERIDO"')

    df.replace(mapping,inplace=True)
    
    # Generate a "clean data file"
    filename = "preprocessing/preprocessed_data.csv"
    df.to_csv(filename,index=False)
    
    # Create a new artifact and upload
    task.upload_artifact(name=args.artifact_name, artifact_object=filename,metadata=dict(
                                    description=args.artifact_description
                                ))
    


    # Remote temporary files
    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--input_artifact",
        default="raw_data",
        type=str,
        help="Fully-qualified name for the input artifact"
    )

    parser.add_argument(
        "--artifact_name",default="processed_data", type=str, help="Name for the artifact"
    )


    parser.add_argument(
        "--artifact_description",
        type=str,
        default="Processed Data",
        help="Description for the artifact"
            )

    parser.add_argument(
        "--task_id",
        type=str,
        default="4eb7af98b1d94e01a0314db4426bcae3",
        help="ClearML projectID"

    )

    # get arguments
    ARGS = parser.parse_args()

    task = Task.init(project_name="a ML example",task_name="Prepreocessing")


    # process the arguments
    process_args(ARGS,task)