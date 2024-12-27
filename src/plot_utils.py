import pandas as pd
import numpy as np
import os, sys

# Plottings
import seaborn as sns
from matplotlib import pyplot as plt

def calculate_statistics(df: pd.DataFrame):

    # Revenue em risco e normalizado
    cbk_risk = df.query("CBK == 'Sim'")['Valor'].sum()
    all_revenue = df['Valor'].sum()

    # Qtd de operações cbk
    cbk_oper = df.query("CBK == 'Sim'").shape[0]
    all_oper = df.shape[0]

    print('##################################################################################')
    print(f'O total de revenue em risco devido a cbk é {cbk_risk}')
    print(f'A taxa de risco de cbk desse e-commerce é {round((cbk_risk/all_revenue)*100,2)}%')
    print(f'O revenue garantido desse e-commerce no mes de maio é {all_revenue-cbk_risk}')
    print(f'CBK Rate desse e-commerce {round((cbk_oper/all_oper)*100,2)}%')
    print(f'Qtd de operações de CBK {cbk_oper}')
    print('##################################################################################')



def plot_ecdf_geral(df: pd.DataFrame, col: str):
    """
    Plota o histograma geral da coluna referenciada em col
    """
    plt.figure(figsize=(12,8))
    sns.ecdfplot(df[col], linewidth=2.5, color='green')
    plt.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5)
    plt.xlim(0,2000)
    plt.title(f'ECDF custo mensal de cada cartão {col}')
    plt.ylabel('ECDF')
    plt.xlabel(col)
    plt.show()

def plot_multiple_ecdf(df: pd.DataFrame, plotting_var, hue_var):
    plt.figure(figsize=(16,12))
    sns.ecdfplot(data=df, x=plotting_var, hue=hue_var,linewidth=2.5)
    plt.title(f'ECDF do valor da compra quebrando por turnos')
    

def plot_box_plot(df: pd.DataFrame, col: str):
    """
    Plotar o boxplot em busca de analisar a mediana, Q1 e Q3 e outliers.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot boxplot
    ax = sns.boxplot(x=df[col], orient='v', color='green')
    
    q1 = np.percentile(df[col], 25)
    q3 = np.percentile(df[col], 75)
    median = np.percentile(df[col], 50)

    plt.xlim(0, 2000)
    print(f'Q1: {q1} | Mediana: {median} | Q3: {q3}')

    plt.title(f'Boxplot custo mensal de cada cartão {col}')
    plt.show()

def plot_box_plot_hue(df: pd.DataFrame, col: str, y: str):
    """
    Plotar o boxplot em busca de analisar a mediana, Q1 e Q3 e outliers.
    """
    plt.figure(figsize=(16, 10))
    
    # Plot boxplot
    ax = sns.boxplot(x=col, y=y, data=df)
    plt.title(f'Boxplot da coluna {y} sem agrupamento')
    plt.show()

def calculate_ticket_medio(df: pd.DataFrame):
    valor = df['Valor'].mean()
    var = df['Valor'].std()
    print('########################################################################')
    print(f'O ticket médio por operação é {valor} com uma desvio padrão {var}')
    print('########################################################################')

def plot_count_opers(df: pd.DataFrame):
    df_complete = df.groupby(['Cartão'])['Valor'].count().reset_index()

    # Criando o dataframe
    df_complete['QtdOpers'] = pd.cut(
           df_complete['Valor'], 
           bins=[1,2,4,6,8,500], 
           labels=['1-2','2-4','4-6','6-8','>8']
    )

    # plotando
    plt.figure(figsize=(12,8))
    sns.countplot(data=df_complete, x='QtdOpers',palette='crest')
    plt.title('Contagem do tipo de recorrencias de operações no e-commerce')
    plt.xlabel('Contagem de operações em maio para o mesmo cartão')
    plt.ylabel('Contagem')
    plt.show()

def plot_count_shift(df: pd.DataFrame):
    plt.figure(figsize=(12,8))
    sns.countplot(data=df, x='shift', palette='crest')
    plt.title('Countplot da contagem de operações por turno')
    plt.xlabel('Turnos')
    plt.ylabel('Contagem')
    plt.show()

def plot_heatmap(df: pd.DataFrame, index, columns, values):
    df_plot = df.pivot(index=index, columns=columns, values=values)

    # Plotando o heatmap
    plt.figure(figsize=(22,16))
    sns.heatmap(df_plot, annot=True, linewidths=0.5, cmap='coolwarm', fmt=".2f")
    plt.title(f"Heatmap | Analisando a coluna {values}")
    plt.xlabel(f"Coluna {columns}")
    plt.ylabel(f"Coluna {index}")

def plot_barplot_agg(df: pd.DataFrame, x, y):
    plt.figure(figsize=(16,8))
    sns.barplot(data=df, x=x, y=y)
    plt.title(f"Barplot | Analisando a coluna {y}")
    plt.show() 

def plot_everyday_sells(df: pd.DataFrame, y, stringer):

    average_valor = df[y].mean()

    plt.figure(figsize=(16,10))
    sns.lineplot(data=df,
                x='Dia',
                y='Valor',
                linewidth=2.0)
    plt.title(f'Checando a replicação de vendas | {stringer}')
    plt.xticks(rotation=90)

    plt.axhline(y=average_valor, color='red', linestyle='--', label=f'Média ({average_valor:.2f})')
    plt.show()