import pandas as pd
import numpy as np

def download_merge_data(path):
    url_discentes = 'https://dados.ufrn.br/dataset/80b1a8e9-2e40-4c6c-97ea-d595a3c8b8f5/resource/0e287fe5-badb-4b34-b1bf-8815db5dfbeb/download/d'
    global discentes 
    discentes = pd.read_table(url_discentes, sep=';',dtype=object)
    url_curso = 'https://dados.ufrn.br/dataset/02526b96-cf40-4507-90b0-3afe5ddd53e7/resource/a10bc434-9a2d-491a-ae8c-41cf643c35bc/download/cursos-de-graduacao.csv'
    global dados_curso
    dados_curso = pd.read_csv(url_curso,sep=';',dtype=object)
    url_20192 = 'https://dados.ufrn.br/dataset/8e0cb3ac-b6fa-48ef-a1ee-f2df0b893b72/resource/a947133d-ad21-4907-af12-8f0a91135af4/download/dados-socio-economicos-discentes-2019.2.csv'
    global socio_Economico_20192
    socio_Economico_20192 = pd.read_csv(url_20192,sep=';',dtype=object)
    url_202015 = 'https://dados.ufrn.br/dataset/8e0cb3ac-b6fa-48ef-a1ee-f2df0b893b72/resource/cf818b90-0867-4d2f-a5db-9c4707db34b7/download/dados-socio-economicos-discentes-2020.1.csv'
    global socio_Economico_202015
    socio_Economico_202015 = pd.read_csv(url_202015,sep=';',dtype=object)
    matriculas_20192_url = 'https://dados.ufrn.br/dataset/c8650d55-3c5a-4787-a126-d28a4ef902a6/resource/0d573a4f-de65-4c3d-b6bb-337473bc4e44/download/matricula-componente-20192.csv'
    global matriculas_20192
    matriculas_20192 = pd.read_csv(matriculas_20192_url,sep=';',dtype=object)
    matriculas_20192 = matriculas_20192.loc[:,['id_turma','discente','reposicao','descricao','numero_total_faltas']]
    matriculas_20192.rename(columns={'discente':'id_discente'},inplace=True)
    matriculas_20205_url = 'https://dados.ufrn.br/dataset/c8650d55-3c5a-4787-a126-d28a4ef902a6/resource/54683a60-b998-4933-a4ee-a6331eba8826/download/matricula-componente-20205.csv'
    global matriculas_20205
    matriculas_20205 = pd.read_csv(matriculas_20205_url,sep=';',dtype=object)
    matriculas_20205 = matriculas_20205.loc[:,['id_turma','discente','reposicao','descricao','numero_total_faltas']]
    matriculas_20205.rename(columns={'discente':'id_discente'},inplace=True)
    columns_dis = ['id_discente', 'sexo', 'ano_nascimento', 'raca', 'estado_origem',
       'cidade_origem', 'estado', 'municipio','nivel_ensino','ano_ingresso', 'periodo_ingresso', 'cotista',
       'curso', 'tipo_cota', 'descricao_tipo_cota']
    discentes = discentes.loc[:,columns_dis]
    columns_socioeco = ['id_discente', 'ano', 'periodo', 'renda',
       'possui_bolsa_pesquisa', 'possui_auxilio_alimentacao',
       'possui_auxilio_transporte', 'possui_auxilio_residencia_moradia',
       'id_curso']
    socio_Economico_202015 = socio_Economico_202015.loc[:,columns_socioeco]
    socio_Economico_20192 = socio_Economico_20192.loc[:,columns_socioeco]
    columns_curso = ['id_curso','area_conhecimento']
    dados_curso = dados_curso.loc[:,columns_curso]
    dados_discentes_202015 = pd.merge(discentes,socio_Economico_202015,how='inner',on='id_discente')
    dados_discentes_202015 = pd.merge(dados_discentes_202015,dados_curso,how='left',on='id_curso')
    dados_discentes_202015 = pd.merge(dados_discentes_202015,matriculas_20205,how='inner',on='id_discente')
    dados_discentes_202015.drop_duplicates(inplace=True)
    dados_discentes_20192 = pd.merge(discentes,socio_Economico_20192,how='inner',on='id_discente')
    dados_discentes_20192 = pd.merge(dados_discentes_20192,dados_curso,how='left',on='id_curso')
    dados_discentes_20192 = pd.merge(dados_discentes_20192,matriculas_20192,how='left',on='id_discente')
    dados_discentes_20192.drop_duplicates(inplace=True)
    dados_discentes_20192.to_csv(path + "/dados_merge_discentes_20192.csv",index=False,encoding='utf-8')
    dados_discentes_202015.to_csv(path + "/dados_merge_discentes_20201_5.csv",index=False,encoding='utf-8')

