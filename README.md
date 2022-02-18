# Mlops_student_freedata_analysis_pipeline
---
 Este repositório diz respeito a um projeto de mlops com o intuito de pegar e 
 relacionar os dados abertos da UFRN no intuito de estudar caracteríscticas sócio-economicas 
 do estudante que possam ter interferido no seu desempenho, durante o primeiro semestre remoto 
 após o início da Pandemia do COVID-19 no Brasil em 2020.
 
 ---
 ## 0. Observações importantes
 ---
 Este projeto foi desenvolvido para a avaliação da disciplina DCA0305 - mlops, e tem como principal 
 intuito a criação do pipeline de machine learning passando por todas as etapas:
  1. Download
  2. Checagem dos dados (data_checks)
  3. Preprocessamento (preprocessing)
  4. Segregação dos dados (Segregate)
  5. Treinamento do modelo (model_training | Model)
  
  É importante notar que não há uma etapa de avaliação pós treinamento(evaluate), Para isso foi gerado apenas a 
  matriz de confusão dentro do treinamento do modelo gerando um mapa de calor no seaborn normalizado que
  exibe a matriz com valores no intervalo [0 , 1]. É possível notar outras métricas dentro dos notebooks
  como precisão, acurácia, f1-score, precisão média. 
  
  Outro ponto a se notar diz respeito ao formato de arquivos encontrados no projeto. os diretórios: Download , 
  Model, Segregate; possuem notebooks com o rascunho do processo, cada notebook é estruturado somente no intuito 
  de prototipar o conceito inicial da etapa e portanto são arquivos funcionais entretanto não são estruturados para
  ambiente de produção. Todos os diretório presentes nesse repositório que foram listados como etapas do mlops possuem 
  arquivos ".py" que representam o verdadeiro a tarefa do pipeline de maneira organizada e utilizando conceitos de 
  código limpo. 
  
  As principais bibliotecas utilizadas podem ser vistas no arquivo "requirements.txt"
  O pipeline das tarefas estão escritas como um todo no arquivo "pipeline.py" e para a criação do pipeline foi utilizada
  a ferramenta de mlops ["clearml"](https://clear.ml/) 
  ---
 
 ## 1. Download
 ---
 O download dos dados é feito por link direto com os [dados abertos da UFRN](https://dados.ufrn.br/), os links estão escritos
 no próprio arquivo download.py e dizem respeito aos seguintes datasets:
  1.[Dados sociais dos discentes]('https://dados.ufrn.br/dataset/80b1a8e9-2e40-4c6c-97ea-d595a3c8b8f5/resource/0e287fe5-badb-4b34-b1bf-8815db5dfbeb/download/d')
  2.[Dados dos cursos]('https://dados.ufrn.br/dataset/02526b96-cf40-4507-90b0-3afe5ddd53e7/resource/a10bc434-9a2d-491a-ae8c-41cf643c35bc/download/cursos-de-graduacao.csv')
  3.[Dados socioeconomicos do semestre 20201]('https://dados.ufrn.br/dataset/8e0cb3ac-b6fa-48ef-a1ee-f2df0b893b72/resource/a947133d-ad21-4907-af12-8f0a91135af4/download/dados-socio-economicos-discentes-2019.2.csv')
  4. [matriculas do semestre 2020.5]('https://dados.ufrn.br/dataset/c8650d55-3c5a-4787-a126-d28a4ef902a6/resource/54683a60-b998-4933-a4ee-a6331eba8826/download/matricula-componente-20205.csv')
  Dos dados sociais dos discentes foram retidas informações gerais do discente e seu status dentro da instituição, dos dados 
  do curso vem a área de conhecimento do discente por se tratar de um campo categórico menor que o total d cursos, e dos dados 
  socioeconomicos vêm as informações de bolsa e renda os dados usados para esse último foram de 2020.1 por se tratar do semestre 
  imediatamente anterior ao 2020.5 e este último não possuir estes dados disponíveis no momento que o projeto foi desenvolvido.
  
  Foi feita a junção de dados para um único dataset com as seguintes features:
(id_discente,sexo,ano_nascimento,raca,estado_origem,cidade_origem,
estado,municipio,nivel_ensino,ano_ingresso,
periodo_ingresso,cotista,curso,tipo_cota,
descricao_tipo_cota,ano,periodo,renda,
possui_bolsa_pesquisa,possui_auxilio_alimentacao,
possui_auxilio_transporte,possui_auxilio_residencia_moradia,
id_curso,area_conhecimento,id_turma,reposicao,descricao,numero_total_faltas)

Os dados gerados durante o Download estão disponíveis no diretório Dados.
---
## 2. data_checks
---
Essa etapa diz respeito à integridade dos dados em relações gerais e estrutrais com métricas rápidas.
---
## 3. preprocessing
---
Esta etapa faz transformações nos dados com o intuíto de selecionar e adaptar as colunas para o modelo
de aprendizado de máquina que será implementado posteriormente. As principais transformações realizadas 
é a binarização da coluna alvo simplificando as classes somente em "APROVADO" e "FALHOU" para simplificar
o problema de classificação proposto. Além disso foi feita uma seleção dentre as colunas apresentadas no 
download, as colunas que serão empregues no modelo são as seguintes:
  1. 'sexo': Diz respeito à identificação sexual do discente | Campo categórico 
  2. 'ano_nascimento' : o ano de nascimento do discente | Campo numérico 
  3. 'raca' : Diz respeito à "raça" com a qual o discente se identifica | Campo categórico
  4. 'estado_origem': estado de origem do discente | Campo categórico
  5. 'cidade_origem': Cidade de origem do discente | Campo categórico
  6. 'renda': renda familiar declarada pelo discente | Campo numérico
  7. 'area_conhecimento': Área de conhecimento do curso ao qual o discente pertence | Campo categórico 
  8. 'possui_auxilio_alimentacao': informa se o discente possui auxilio alimentação |Campo binário  
  9. 'possui_auxilio_transporte': informa se o discente possui auxilio transporte |Campo binário
  10. 'possui_auxilio_residencia_moradia': informa se o discente possui auxilio moradia |Campo binário
  11. 'grau_academico': grau acadêmico do curso |Campo categórico
  12. 'descricao': informa se o discente foi aprovado ou reprovado no semestre estudado |Campo binário
  


