import streamlit as st
import pandas as pd
import numpy as np

# Título da aplicação
st.title("Backtest de Estratégias de Apostas")

# Função genérica de Backtest
def run_backtest(df, estrategia_func, estrategia_nome):
    df_filtrado = estrategia_func(df)
    df_filtrado['Profit'] = df_filtrado.apply(
        lambda row: (row['Odd_H_Back'] - 1) if row['Goals_H'] > row['Goals_A'] else -1,
        axis=1
    )
    total_jogos = len(df_filtrado)
    acertos = len(df_filtrado[df_filtrado['Goals_H'] > df_filtrado['Goals_A']])
    taxa_acerto = acertos / total_jogos if total_jogos > 0 else 0
    lucro_total = df_filtrado['Profit'].sum()
    
    return {
        "Estratégia": estrategia_nome,
        "Total de Jogos": total_jogos,
        "Taxa de Acerto": f"{taxa_acerto:.2%}",
        "Lucro Total": f"{lucro_total:.2f}",
        "Dataframe": df_filtrado
    }

# Análise das médias
def check_moving_averages(df_filtrado, estrategia_nome):
    df_filtrado['Acerto'] = (df_filtrado['Goals_H'] > df_filtrado['Goals_A']).astype(int)
    ultimos_8 = df_filtrado.tail(8) if len(df_filtrado) >= 8 else df_filtrado
    ultimos_40 = df_filtrado.tail(40) if len(df_filtrado) >= 40 else df_filtrado
    media_8 = ultimos_8['Acerto'].sum() / 8 if len(ultimos_8) == 8 else ultimos_8['Acerto'].mean()
    media_40 = ultimos_40['Acerto'].sum() / 40 if len(ultimos_40) == 40 else ultimos_40['Acerto'].mean()
    acima_das_medias = media_8 > 0.5 and media_40 > 0.5
    
    return {
        "Estratégia": estrategia_nome,
        "Média 8": f"{media_8:.2f} ({ultimos_8['Acerto'].sum()} acertos em {len(ultimos_8)})",
        "Média 40": f"{media_40:.2f} ({ultimos_40['Acerto'].sum()} acertos em {len(ultimos_40)})",
        "Acima dos Limiares": acima_das_medias
    }

# Analisar jogos do dia
def analyze_daily_games(df_daily, estrategia_func, estrategia_nome):
    df_filtrado = estrategia_func(df_daily)
    if not df_filtrado.empty:
        return df_filtrado[['Time', 'Home', 'Away']]
    return None

# Pre-calcular variáveis
def pre_calculate_all_vars(df):
    probs = {
        'pH': 1 / df['Odd_H_Back'],
        'pD': 1 / df['Odd_D_Back'],
        'pA': 1 / df['Odd_A_Back'],
        'pOver': 1 / df['Odd_Over25_FT_Back'],
        'pUnder': 1 / df['Odd_Under25_FT_Back'],
        'pBTTS_Y': 1 / df['Odd_BTTS_Yes_Back'],
        'pBTTS_N': 1 / df['Odd_BTTS_No_Back'],
        'p0x0': 1 / df['Odd_CS_0x0_Lay'],
        'p0x1': 1 / df['Odd_CS_0x1_Lay'],
        'p1x0': 1 / df['Odd_CS_1x0_Lay']
    }
    
    vars_dict = {
        'VAR01': probs['pH'] / probs['pD'],
        'VAR02': probs['pH'] / probs['pA'],
        'VAR03': probs['pD'] / probs['pH'],
        'VAR04': probs['pD'] / probs['pA'],
        'VAR05': probs['pA'] / probs['pH'],
        'VAR06': probs['pA'] / probs['pD'],
        'VAR07': probs['pOver'] / probs['pUnder'],
        'VAR08': probs['pUnder'] / probs['pOver'],
        'VAR09': probs['pBTTS_Y'] / probs['pBTTS_N'],
        'VAR10': probs['pBTTS_N'] / probs['pBTTS_Y'],
        'VAR11': probs['pH'] / probs['pOver'],
        'VAR12': probs['pD'] / probs['pOver'],
        'VAR13': probs['pA'] / probs['pOver'],
        'VAR14': probs['pH'] / probs['pUnder'],
        'VAR15': probs['pD'] / probs['pUnder'],
        'VAR16': probs['pA'] / probs['pUnder'],
        'VAR17': probs['pH'] / probs['pBTTS_Y'],
        'VAR18': probs['pD'] / probs['pBTTS_Y'],
        'VAR19': probs['pA'] / probs['pBTTS_Y'],
        'VAR20': probs['pH'] / probs['pBTTS_N'],
        'VAR21': probs['pD'] / probs['pBTTS_N'],
        'VAR22': probs['pA'] / probs['pBTTS_N'],
        'VAR23': probs['p0x0'] / probs['pH'],
        'VAR24': probs['p0x0'] / probs['pD'],
        'VAR25': probs['p0x0'] / probs['pA'],
        'VAR26': probs['p0x0'] / probs['pOver'],
        'VAR27': probs['p0x0'] / probs['pUnder'],
        'VAR28': probs['p0x0'] / probs['pBTTS_Y'],
        'VAR29': probs['p0x0'] / probs['pBTTS_N'],
        'VAR30': probs['p0x1'] / probs['pH'],
        'VAR31': probs['p0x1'] / probs['pD'],
        'VAR32': probs['p0x1'] / probs['pA'],
        'VAR33': probs['p0x1'] / probs['pOver'],
        'VAR34': probs['p0x1'] / probs['pUnder'],
        'VAR35': probs['p0x1'] / probs['pBTTS_Y'],
        'VAR36': probs['p0x1'] / probs['pBTTS_N'],
        'VAR37': probs['p1x0'] / probs['pH'],
        'VAR38': probs['p1x0'] / probs['pD'],
        'VAR39': probs['p1x0'] / probs['pA'],
        'VAR40': probs['p1x0'] / probs['pOver'],
        'VAR41': probs['p1x0'] / probs['pUnder'],
        'VAR42': probs['p1x0'] / probs['pBTTS_Y'],
        'VAR43': probs['p1x0'] / probs['pBTTS_N'],
        'VAR44': probs['p0x0'] / probs['p0x1'],
        'VAR45': probs['p0x0'] / probs['p1x0'],
        'VAR46': probs['p0x1'] / probs['p0x0'],
        'VAR47': probs['p0x1'] / probs['p1x0'],
        'VAR48': probs['p1x0'] / probs['p0x0'],
        'VAR49': probs['p1x0'] / probs['p0x1'],
        'VAR50': (probs['pH'].to_frame().join(probs['pD'].to_frame()).join(probs['pA'].to_frame())).std(axis=1) /
                 (probs['pH'].to_frame().join(probs['pD'].to_frame()).join(probs['pA'].to_frame())).mean(axis=1),
        'VAR51': (probs['pOver'].to_frame().join(probs['pUnder'].to_frame())).std(axis=1) /
                 (probs['pOver'].to_frame().join(probs['pUnder'].to_frame())).mean(axis=1),
        'VAR52': (probs['pBTTS_Y'].to_frame().join(probs['pBTTS_N'].to_frame())).std(axis=1) /
                 (probs['pBTTS_Y'].to_frame().join(probs['pBTTS_N'].to_frame())).mean(axis=1),
        'VAR53': (probs['p0x0'].to_frame().join(probs['p0x1'].to_frame()).join(probs['p1x0'].to_frame())).std(axis=1) /
                 (probs['p0x0'].to_frame().join(probs['p0x1'].to_frame()).join(probs['p1x0'].to_frame())).mean(axis=1),
        'VAR54': abs(probs['pH'] - probs['pA']),
        'VAR55': abs(probs['pH'] - probs['pD']),
        'VAR56': abs(probs['pD'] - probs['pA']),
        'VAR57': abs(probs['pOver'] - probs['pUnder']),
        'VAR58': abs(probs['pBTTS_Y'] - probs['pBTTS_N']),
        'VAR59': abs(probs['p0x0'] - probs['p0x1']),
        'VAR60': abs(probs['p0x0'] - probs['p1x0']),
        'VAR61': abs(probs['p0x1'] - probs['p1x0']),
        'VAR62': np.arctan((probs['pA'] - probs['pH']) / 2) * 180 / np.pi,
        'VAR63': np.arctan((probs['pD'] - probs['pH']) / 2) * 180 / np.pi,
        'VAR64': np.arctan((probs['pA'] - probs['pD']) / 2) * 180 / np.pi,
        'VAR65': np.arctan((probs['pUnder'] - probs['pOver']) / 2) * 180 / np.pi,
        'VAR66': np.arctan((probs['pBTTS_N'] - probs['pBTTS_Y']) / 2) * 180 / np.pi,
        'VAR67': np.arctan((probs['p0x1'] - probs['p0x0']) / 2) * 180 / np.pi,
        'VAR68': np.arctan((probs['p1x0'] - probs['p0x0']) / 2) * 180 / np.pi,
        'VAR69': np.arctan((probs['p1x0'] - probs['p0x1']) / 2) * 180 / np.pi,
        'VAR70': abs(probs['pH'] - probs['pA']) / probs['pA'],
        'VAR71': abs(probs['pH'] - probs['pD']) / probs['pD'],
        'VAR72': abs(probs['pD'] - probs['pA']) / probs['pA'],
        'VAR73': abs(probs['pOver'] - probs['pUnder']) / probs['pUnder'],
        'VAR74': abs(probs['pBTTS_Y'] - probs['pBTTS_N']) / probs['pBTTS_N'],
        'VAR75': abs(probs['p0x0'] - probs['p0x1']) / probs['p0x1'],
        'VAR76': abs(probs['p0x0'] - probs['p1x0']) / probs['p1x0'],
        'VAR77': abs(probs['p0x1'] - probs['p1x0']) / probs['p1x0']
    }
    return vars_dict

# Definição das estratégias
def apply_strategies(df):
    vars_dict = pre_calculate_all_vars(df)
    
    def estrategia_1(df): return df[(vars_dict['VAR01'] >= 1.5) & (vars_dict['VAR01'] <= 3.0) & (vars_dict['VAR07'] >= 0.8) & (vars_dict['VAR07'] <= 1.2)].copy()
    def estrategia_2(df): return df[(vars_dict['VAR02'] >= 0.5) & (vars_dict['VAR02'] <= 2.0) & (vars_dict['VAR17'] >= 0.8) & (vars_dict['VAR17'] <= 1.5)].copy()
    def estrategia_3(df): return df[(vars_dict['VAR11'] >= 0.7) & (vars_dict['VAR11'] <= 1.3) & (vars_dict['VAR23'] >= 0.2) & (vars_dict['VAR23'] <= 0.6)].copy()
    def estrategia_4(df): return df[(vars_dict['VAR37'] >= 0.3) & (vars_dict['VAR37'] <= 0.7) & (vars_dict['VAR43'] >= 0.4) & (vars_dict['VAR43'] <= 0.9)].copy()
    def estrategia_5(df): return df[(vars_dict['VAR54'] >= 0.1) & (vars_dict['VAR54'] <= 0.4) & (vars_dict['VAR57'] >= 0.05) & (vars_dict['VAR57'] <= 0.3)].copy()
    def estrategia_6(df): return df[(vars_dict['VAR62'] >= -20) & (vars_dict['VAR62'] <= 20) & (vars_dict['VAR65'] >= -15) & (vars_dict['VAR65'] <= 15)].copy()
    def estrategia_7(df): return df[(vars_dict['VAR70'] >= 0.2) & (vars_dict['VAR70'] <= 0.8) & (vars_dict['VAR74'] >= 0.1) & (vars_dict['VAR74'] <= 0.5)].copy()
    def estrategia_8(df): return df[(vars_dict['VAR45'] >= 0.5) & (vars_dict['VAR45'] <= 1.5) & (vars_dict['VAR47'] >= 0.4) & (vars_dict['VAR47'] <= 1.2)].copy()
    def estrategia_9(df): return df[(vars_dict['VAR50'] >= 0.1) & (vars_dict['VAR50'] <= 0.4) & (vars_dict['VAR52'] >= 0.05) & (vars_dict['VAR52'] <= 0.3)].copy()
    def estrategia_10(df): return df[(vars_dict['VAR33'] >= 0.2) & (vars_dict['VAR33'] <= 0.6) & (vars_dict['VAR77'] >= 0.1) & (vars_dict['VAR77'] <= 0.5)].copy()

    return [
        (estrategia_1, "Estratégia 1"), (estrategia_2, "Estratégia 2"), (estrategia_3, "Estratégia 3"),
        (estrategia_4, "Estratégia 4"), (estrategia_5, "Estratégia 5"), (estrategia_6, "Estratégia 6"),
        (estrategia_7, "Estratégia 7"), (estrategia_8, "Estratégia 8"), (estrategia_9, "Estratégia 9"),
        (estrategia_10, "Estratégia 10")
    ]

# Interface Streamlit
st.header("Upload da Planilha Histórica")
uploaded_historical = st.file_uploader("Faça upload da planilha histórica (xlsx)", type=["xlsx"])

if uploaded_historical is not None:
    df_historico = pd.read_excel(uploaded_historical)
    estrategias = apply_strategies(df_historico)
    
    # Executar backtest
    st.header("Resultados do Backtest")
    backtest_results = []
    medias_results = []
    resultados = {}
    
    for estrategia_func, estrategia_nome in estrategias:
        backtest_result = run_backtest(df_historico, estrategia_func, estrategia_nome)
        medias_result = check_moving_averages(backtest_result["Dataframe"], estrategia_nome)
        backtest_results.append(backtest_result)
        medias_results.append(medias_result)
        resultados[estrategia_nome] = (backtest_result["Dataframe"], medias_result["Acima dos Limiares"])
    
    # Exibir resultados do backtest
    st.subheader("Resumo do Backtest")
    st.dataframe(pd.DataFrame([r for r in backtest_results if r["Total de Jogos"] > 0]).drop(columns=["Dataframe"]))
    
    # Exibir análise das médias
    st.subheader("Análise das Médias")
    st.dataframe(pd.DataFrame(medias_results))
    
    # Upload dos jogos do dia para estratégias aprovadas
    estrategias_aprovadas = [nome for nome, (_, acima) in resultados.items() if acima]
    if estrategias_aprovadas:
        st.header("Upload dos Jogos do Dia")
        uploaded_daily = st.file_uploader("Faça upload da planilha com os jogos do dia (xlsx)", type=["xlsx"])
        
        if uploaded_daily is not None:
            df_daily = pd.read_excel(uploaded_daily)
            st.header("Jogos Aprovados para Hoje")
            
            for estrategia_nome in estrategias_aprovadas:
                estrategia_func = next(func for func, nome in estrategias if nome == estrategia_nome)
                jogos_aprovados = analyze_daily_games(df_daily, estrategia_func, estrategia_nome)
                if jogos_aprovados is not None:
                    st.subheader(f"{estrategia_nome}")
                    st.dataframe(jogos_aprovados)
                else:
                    st.write(f"Nenhum jogo do dia atende aos critérios da {estrategia_nome}.")
    else:
        st.write("Nenhuma estratégia passou na análise das médias.")
else:
    st.write("Por favor, faça upload da planilha histórica para começar.")