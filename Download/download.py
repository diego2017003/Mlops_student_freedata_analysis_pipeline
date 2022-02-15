"""
Creator: Reyne
Date: 10 Feb. 2022
Download the raw data from dados.ufrn.br
"""
import argparse
import logging
import requests
import tempfile
import pandas as pd

from clearml import Task

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')





def process_args(args,URLS,task):
    # Download file, streaming so we can download files larger than
    # the available memory. We use a named temporary file that gets
    # destroyed at the end of the context, so we don't leave anything
    # behind and the file gets removed even in case of errors



    #logger.info(f"Downloading {args.file_url} ...")
    logger.report_text("Downloading Dataframes ...",level=logging.INFO)

    dados_curso = pd.read_csv(URLS['cursos'],sep=';',dtype=object)
    discentes = pd.read_csv(URLS['discentes'],sep=';',dtype=object)

    matriculas_20205 = pd.read_csv(URLS['matriculas_20205'],sep=';',dtype=object)
    matriculas_20205 = matriculas_20205.loc[:,['id_turma','discente','reposicao','descricao','numero_total_faltas']]
    matriculas_20205.rename(columns={'discente':'id_discente'},inplace=True)

    columns_dis = ['id_discente', 'sexo', 'ano_nascimento', 'raca', 'estado_origem',
       'cidade_origem', 'estado', 'municipio','nivel_ensino','ano_ingresso', 
       'curso']

    discentes = discentes.loc[:,columns_dis]

    columns_socioeco = ['id_discente', 'ano', 'periodo', 'renda',
       'possui_bolsa_pesquisa', 'possui_auxilio_alimentacao',
       'possui_auxilio_transporte', 'possui_auxilio_residencia_moradia',
       'id_curso'] 

    
    socio_Economico_202015 = pd.read_csv(URLS['socioeconomicos_20205'],sep=';',dtype=object)

    socio_Economico_202015 = socio_Economico_202015.loc[:,columns_socioeco]

    columns_curso = ['id_curso','area_conhecimento']

    

    dados_discentes_202015 = pd.merge(discentes,socio_Economico_202015,how='inner',on='id_discente')
    dados_discentes_202015 = pd.merge(dados_discentes_202015,dados_curso,how='left',on='id_curso')
    dados_discentes_202015 = pd.merge(dados_discentes_202015,matriculas_20205,how='inner',on='id_discente')


    with tempfile.NamedTemporaryFile(mode='wb+') as fp:

        logger.report_text("Sending to ClearML...")

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            dados_discentes_202015.to_csv(temp.name + '.csv')
            

            task.upload_artifact(name=args.artifact_name, artifact_object=temp.name + '.csv',metadata=dict(
                                    description=args.artifact_description
                                ))

            temp.close()



            # Make sure the file has been written to disk before uploading
            # to W&B
            fp.flush()

            logger.report_text("artifact Uploaded")



if __name__ == "__main__":

    URLS = dict(

        discentes = 'https://dados.ufrn.br/dataset/80b1a8e9-2e40-4c6c-97ea-d595a3c8b8f5/resource/0e287fe5-badb-4b34-b1bf-8815db5dfbeb/download/d',
        cursos = 'https://dados.ufrn.br/dataset/02526b96-cf40-4507-90b0-3afe5ddd53e7/resource/a10bc434-9a2d-491a-ae8c-41cf643c35bc/download/cursos-de-graduacao.csv',
        socioeconomicos_20192= 'https://dados.ufrn.br/dataset/8e0cb3ac-b6fa-48ef-a1ee-f2df0b893b72/resource/a947133d-ad21-4907-af12-8f0a91135af4/download/dados-socio-economicos-discentes-2019.2.csv',
        matriculas_20205 = 'https://dados.ufrn.br/dataset/c8650d55-3c5a-4787-a126-d28a4ef902a6/resource/54683a60-b998-4933-a4ee-a6331eba8826/download/matricula-componente-20205.csv',
        socioeconomicos_20205 = "https://dados.ufrn.br/dataset/8e0cb3ac-b6fa-48ef-a1ee-f2df0b893b72/resource/cf818b90-0867-4d2f-a5db-9c4707db34b7/download/dados-socio-economicos-discentes-2020.1.csv"

    )



    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to ClearML", fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--artifact_name", type=str,default="raw_data", help="Name for the artifact"
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        default="Data from UFRN",
        help="Description for the artifact"
    )

    ARGS = parser.parse_args()

    task = Task.init(project_name="a ML example",task_name="Download Raw data")
    task.connect(URLS)

    logger = task.get_logger()

    process_args(ARGS,URLS,task)
