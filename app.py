from joblib import load
import gradio as gr
import pandas as pd

from  model_training.log_regress import *

clf = load("model_training/regressao_logistica.joblib")

def passou_ou_nao(sexo, ano_nasc, raca, renda, auxilios, grau,area, da_capital):

    auxilios_list=3*["False"]

    for i in auxilios:
        #print(i)
        auxilios_list[i] = "True"

    aux_ali, aux_Mor, aux_Trans = auxilios_list
    entrada = [[sexo[0],ano_nasc,raca,renda, aux_ali, aux_Trans, aux_Mor, grau,area,da_capital]]

    headers = ['sexo', 'ano_nascimento', 'raca', 'renda', 'possui_auxilio_alimentacao',
       'possui_auxilio_transporte', 'possui_auxilio_residencia_moradia',
       'grau_academico', 'area_conhecimento', 'local_ou_de_fora']

    entrada_df = pd.DataFrame(entrada,columns=headers)

    prediction = clf.predict_proba(entrada_df)[0]

    class_names = ["REPROVAR", "APROVADO"]

    return {class_names[i]: prediction[i] for i in range(2)}

#set the user uploaded image as the input array
#match same shape as the input shape in the model

#setup the interface
iface = gr.Interface(
    passou_ou_nao,
    [
        gr.inputs.Radio(["Masculino", "Feminino"]),
        gr.inputs.Number(label="Ano de nascimento"),
        gr.inputs.Radio(['Negro', 'Branco', 'Pardo',
                        'Amarelo (de origem oriental)', 'Indígena',
                        'Remanescente de quilombo']),
        gr.inputs.Number(label="Sua Renda familiar"),
        gr.inputs.CheckboxGroup( ["Alimentação","Moradia/Residência","Transporte"], 
                                         label="Possui Auxilio ... ?",type='index'),
        
        gr.inputs.Dropdown(['BACHARELADO', 'LICENCIATURA', 'TECNOLÓGICO'], label="Nível"),
        gr.inputs.Dropdown(['Ciências Biológicas', 'Ciências da Saúde',
       'Ciências Exatas e da Terra', 'Engenharias',
       'Ciências Sociais Aplicadas', 'Ciências Humanas',
       'Linguística, Letras e Artes', 'Ciências Agrárias', 'Outra'],
       label="Sua Aréa de conhecimento"),
       gr.inputs.Checkbox(default=False, label="Sou originário da capital do RN"),
    ], 
    
    outputs = gr.outputs.Label(),
    interpretation="default"
)

iface.launch()
