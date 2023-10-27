import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image


def load_model(name):
    import pickle
    arq = 'models/' + name + '.pkl'
    with open(arq, 'rb') as file:
        model = pickle.load(file)
    # print('Model sklearn loaded')
    return model


def plot_gauge(prob) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Probabilidade de Evasão (%)"},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%'},
            'bar': {'color': ("#FF7131" if prob < 50 else "#4c1854")},
            'steps': [
                {'range': [0, 25], 'color': "#f2f2f2"},
                {'range': [25, 50], 'color': "#f2f2f2"},
                {'range': [50, 75], 'color': "#f2f2f2"},
                {'range': [75, 100], 'color': "#f2f2f2"},
            ],
        }
    ))
    return fig

def plot_gauge_en(prob) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Dropout Probability (%)"},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%'},
            'bar': {'color': ("#FF7131" if prob < 50 else "#4c1854")},
            'steps': [
                {'range': [0, 25], 'color': "#f2f2f2"},
                {'range': [25, 50], 'color': "#f2f2f2"},
                {'range': [50, 75], 'color': "#f2f2f2"},
                {'range': [75, 100], 'color': "#f2f2f2"},
            ],
        }
    ))
    return fig


st.set_page_config(layout='wide')

cidades = ['Terra Roxa', 'Palotina', 'Assis Chateaubriand', 'Foz do Iguaçu', 'Colíder', 'Nova Santa Rosa',
            'Francisco Alves', 'Curitiba', 'Sapucaia do Sul', 'Maripá', 'Paranaguá', 'Cafelândia',
            'Marechal Cândido Rondon', 'Toledo', 'Cruzeiro do Oeste', 'Iporã', 'Brasília', 'São Paulo', ]

col01, col02 = st.columns([4, 1])
with col01:
    st.title('Predição de Evasão')
    st.text('by: Prof. Dr. Marcos Oliveira (marcos.assis@ufpr.br)')

with col02:
    image = Image.open('src/app/img/liccomp.jpeg')
    st.image(image, width=300 )

aba1, aba2 = st.tabs(['Portuguese-BR', 'English'])

with aba1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Informações')
        col11, col12 = st.columns(2)
        with col11:
            tempoUniversidade = st.number_input('Tempo de Universidade (semestres)', min_value=0, max_value=16)
            anoConclusaoEnsinoMedio = st.number_input('Ano de Conclusão do Ens. Médio (0 se não informado)', min_value=0) #ver range
            ira = st.number_input('IRA', min_value=0, max_value=100)
            city = st.selectbox('Cidade', cidades, index=1)

        with col12:
            rep_freq = st.number_input('Número de Reprovações por Frequência', min_value=0)
            DEE374_freq = st.number_input('Pré-Cálculo (Frequencia média) - 1o sem.', min_value=0, max_value=100)
            DEE341_freq = st.number_input('Laboratório de Programação I (Frequencia média) - 1o sem.', min_value=0, max_value=100)
            DEE345_nota = st.number_input('Segurança de Sistemas Computacionais (Nota média) - 2o sem.', min_value=0, max_value=100)

    with col2:
        # if st.button('Prever'):
        modelo = load_model('xgboost_tuned')
        encoder = load_model('target_encoder')

        dados = pd.DataFrame([[tempoUniversidade, DEE374_freq, anoConclusaoEnsinoMedio,
                              rep_freq, DEE345_nota, ira, city, DEE341_freq]],
                             columns=['tempoUniversidade', 'DEE374_freq',
                                       'anoConclusaoEnsinoMedio', 'Reprovado por frequência', 'DEE345_nota',
                                      'ira', 'cidade', 'DEE341_freq'])

        dados = encoder.transform(dados)

        pred = modelo.predict(dados)[0]
        pred_prob = modelo.predict_proba(dados)[0][1]

        # print(pred)
        # print(pred_prob)

        st.header('RESULTADO:', divider='grey')
        if pred==0:
            st.markdown(f'<h2><font color="#FF7131">Evasão improvável</font></h2>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<h2><font color="#FF7131">Evasão PROVÁVEL</font></h2>',
                        unsafe_allow_html=True)

        fig = plot_gauge(pred_prob * 100)
        st.plotly_chart(fig, use_container_width=True)

with aba2:
    col1en, col2en = st.columns(2)
    with col1en:
        st.subheader('Informations')
        col21, col22 = st.columns(2)
        with col21:
            tempoUniversidade2 = st.number_input('University time (semesters)', min_value=0, max_value=16)
            anoConclusaoEnsinoMedio2 = st.number_input('Year of high school completion (0 if not provided)',
                                                      min_value=0)  # ver range
            ira2 = st.number_input('Academic Performance Index - IRA', min_value=0, max_value=100)
            city2 = st.selectbox('City', cidades, index=1)

        with col22:
            rep_freq2 = st.number_input('Failures due to attendance', min_value=0)
            DEE374_freq2 = st.number_input('PRE-CALCULUS (Mean Attendance) - 1st sem.', min_value=0, max_value=100)
            DEE341_freq2 = st.number_input('PROGRAMMING LABORATORY I (Mean Attendance) - 1st sem.', min_value=0,
                                          max_value=100)
            DEE345_nota2 = st.number_input('COMPUTER SYSTEMS SECURITY (Mean Score) - 2nd sem.', min_value=0,
                                          max_value=100)

    with col2en:
        # if st.button('Prever'):
        modelo = load_model('xgboost_tuned')
        encoder = load_model('target_encoder')

        dados = pd.DataFrame([[tempoUniversidade2, DEE374_freq2, anoConclusaoEnsinoMedio2,
                               rep_freq2, DEE345_nota2, ira2, city2, DEE341_freq2]],
                             columns=['tempoUniversidade', 'DEE374_freq',
                                      'anoConclusaoEnsinoMedio', 'Reprovado por frequência', 'DEE345_nota',
                                      'ira', 'cidade', 'DEE341_freq'])

        dados = encoder.transform(dados)

        pred = modelo.predict(dados)[0]
        pred_prob = modelo.predict_proba(dados)[0][1]

        st.header('RESULT:', divider='grey')
        if pred == 0:
            st.markdown(f'<h2><font color="#FF7131">Unlikely dropout</font></h2>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<h2><font color="#FF7131">PROBABLE DROPOUT</font></h2>',
                        unsafe_allow_html=True)

        fig = plot_gauge_en(pred_prob * 100)
        st.plotly_chart(fig, use_container_width=True)


