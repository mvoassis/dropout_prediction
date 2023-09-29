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


st.set_page_config(layout='wide')

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
            rep_nota = st.number_input('Número de Reprovações por Nota', min_value=0)
            rep_freq = st.number_input('Número de Reprovações por Frequência', min_value=0)
        with col12:
            DEE346_freq = st.number_input('Algoritmos 1 (Freq) - 1o sem.', min_value=0, max_value=100)
            DEE374_freq = st.number_input('Pré-Cálculo (Freq) - 1o sem.', min_value=0, max_value=100)
            # DEE349_freq = st.number_input('Comp e Sociedade (Freq) - 3o sem.', min_value=0, max_value=100)
            DEC008_freq = st.number_input('Libras I (Freq) - 2o sem.', min_value=0, max_value=100)
            # DEE351_nota = st.number_input('Eng. Software (Nota) - 3o sem.', min_value=0, max_value=100)

    with col2:
        # if st.button('Prever'):
        modelo = load_model('gbc')
        dados = pd.DataFrame([[DEE346_freq, tempoUniversidade, DEE374_freq, anoConclusaoEnsinoMedio,
                              rep_freq, ira, DEC008_freq, rep_nota]],
                             columns=['DEE346_freq', 'tempoUniversidade', 'DEE374_freq',
                                       'anoConclusaoEnsinoMedio', 'Reprovado por frequência', 'ira', 'DEC008_freq',
                                       'Reprovado por nota'])

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



