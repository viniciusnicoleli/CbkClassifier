from sklearn.preprocessing import (OneHotEncoder, 
                                   OrdinalEncoder, 
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.metrics import (precision_recall_curve, 
                             PrecisionRecallDisplay, 
                             roc_curve, 
                             RocCurveDisplay, 
                             auc)
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, 
                             average_precision_score, 
                             classification_report, 
                             confusion_matrix,
                             recall_score,
                             precision_score)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from typing import List, Dict, Tuple                   
from matplotlib import pyplot as plt
from IPython.display import display
import seaborn as sns
import pandas as pd
import numpy as np
import os

def pipeline_numerical(numerical_scaler, num_cols):
    if num_cols.empty:
        return None
    return ('pipe_num',Pipeline(
        steps=[
            ("selector_numerical", ColumnTransformer([("filter_num_cols", "passthrough", num_cols.columns.values)], remainder='drop')),
            ("num_imputer", SimpleImputer(strategy='mean')),
            ("NumScaler", numerical_scaler)
        ]
    ))

def pipeline_one_hot(one_hot_cols):
    if one_hot_cols.empty:
        return None
    return ('pipe_hot',Pipeline(
        steps=[
            ("selector_one_hot", ColumnTransformer([("filter_one_cols", "passthrough", one_hot_cols.columns.values)], remainder='drop')),
            ("one_imputer", SimpleImputer(strategy='most_frequent')),
            ("OneHotEncoder", OneHotEncoder(handle_unknown='ignore'))
        ]
    ))

def pipeline_ordinal(ordinal_cols):
    if ordinal_cols.empty:
        return None
    return ('pipe_ord',Pipeline(
        steps=[
            ("selector_one_hot", ColumnTransformer([("filter_one_cols", "passthrough", ordinal_cols.columns.values)], remainder='drop')),
            ("one_imputer", SimpleImputer(strategy='most_frequent')),
            ("OrdinalEncoder", OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
        ]
    ))



def create_pipeline(df: pd.DataFrame, 
                    columns_ignore: List, 
                    ordinal_order: Dict[str,List] = None, 
                    numerical_scaler = StandardScaler()):
    
    #Eliminando do Pipeline colunas que precisam ser ignoradas
    columns_ignore_all = list(ordinal_order.keys()) + columns_ignore if ordinal_order else columns_ignore

    print(f'Ignorando essas colunas tanto para OneHot quanto para Numerical: {columns_ignore_all}')

    # Criando os dataframes que contém as features para transformação
    numerical_features: pd.DataFrame = df[[col for col in df.select_dtypes(include=['float','int']).columns if col not in columns_ignore_all]]
    one_hot_features: pd.DataFrame = df[[col for col in df.select_dtypes(include=['object'], exclude=['datetime']).columns if col not in columns_ignore_all]]
    ordinal_features: pd.DataFrame = df[list(ordinal_order.keys())] if ordinal_order else pd.DataFrame()

    print(f'DataFrames criados sendo numericas:{numerical_features.shape[1]}, one_hot:{one_hot_features.shape[1]}, ordinal:{ordinal_features.shape[1]}')

    # Criando o Pipeline NumScaler
    pipe_num: Tuple = pipeline_numerical(numerical_scaler=numerical_scaler, 
                                         num_cols=numerical_features)
    
    # Criando o Pipeline OneHotEncoder
    pipe_one_hot: Tuple = pipeline_one_hot(one_hot_cols=one_hot_features)
    
    # Criando o Pipeline Ordinal
    pipe_ordinal: Tuple = pipeline_ordinal(ordinal_cols=ordinal_features)

    print(f'Pipelines criados, criando of FeatureUnion')

    return (FeatureUnion(
        transformer_list=[pipe for pipe in [pipe_num, pipe_one_hot, pipe_ordinal] if pipe is not None],
        verbose=True
    ), numerical_features.columns, one_hot_features.columns, ordinal_features.columns)

def create_training_pipeline(pipe_features, columns, model):
    return Pipeline([
        ('transformer_prep', pipe_features),
        ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns = columns))),
        ('estimator', model)
    ])

def plot_rfe_train_scores(pipe) -> Dict:
    rfecv = pipe['estimator']

    print("Número ótimo de features: %d" % rfecv.n_features_)
    print(f"Ranking das features {rfecv.ranking_}")

    explaining = {
        'rows/index':'As rows são as features sendo excluidas recursivamente',
        'colunas':'Possuimos o valor do teste médio assim o resultado individual de cada teste, quanto mais cv mais colunas'
    }

    plt.figure(figsize=(12,6)) 
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], label="Cross-validation score")
    plt.xlabel("Número de features selecionadas")
    plt.ylabel("Cross validation score")
    plt.legend(loc="best")
    plt.show()

    display(pd.DataFrame(rfecv.cv_results_))

    transformed_features = pipe.named_steps['transformer_prep'].get_feature_names_out()
    selected_features = transformed_features[pipe.named_steps['estimator'].support_]

    print('Features que o RFECV recomenda:')
    print(selected_features)

    return explaining, selected_features

def create_columns_listing(columns_selected: list):
    print(columns_selected)
    one_hot_features = list(set([name.split('pipe_hot__filter_one_cols__')[1].split('_')[0] for name in columns_selected if name.startswith('pipe_hot__filter_one_cols__')]))
    num_cols = list(set([name.split('pipe_num__filter_num_cols__')[1] for name in columns_selected if name.startswith('pipe_num__filter_num_cols__')]))
    ord_features = list(set([name.split('pipe_ord__filter_one_cols__')[1].split('_')[0] for name in columns_selected if name.startswith('pipe_ord__filter_one_cols__')]))

    [result for result in [num_cols, one_hot_features, ord_features] if len(result) > 0]
    return list(set(num_cols + one_hot_features + ord_features))

def plot_precision_recall_and_roc(y_true, y_pred, estimator_name="Estimator"):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=estimator_name)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    pr_display.plot(ax=axes[0])
    axes[0].set_title('Precision-Recall Curve')
    axes[0].grid(True)
    
    roc_display.plot(ax=axes[1])
    axes[1].set_title('ROC Curve')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_dist(y_train, pred_proba_train, y_val, pred_proba_val):
    
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (22,12))
    plt.subplots_adjust(left = None, right = None, top = None, bottom = None, wspace = 0.2, hspace = 0.4)
    
    vis = pd.DataFrame()
    vis['target'] = y_train
    vis['proba'] = pred_proba_train
    
    list_1 = vis[vis.target == 1].proba
    list_2 = vis[vis.target == 0].proba
    
    sns.distplot(list_1, kde = True, ax = axs[0], hist = True, bins = 100)
    sns.distplot(list_2, kde = True, ax = axs[0], hist = True, bins = 100)
    
    axs[0].set_title('train Thereshold Curve')
    
    
    
    vis = pd.DataFrame()
    vis['target'] = y_val
    vis['proba'] = pred_proba_val
    
    list_1 = vis[vis.target == 1].proba
    list_2 = vis[vis.target == 0].proba
    
    sns.distplot(list_1, kde = True, ax = axs[1], hist = True, bins = 100)
    sns.distplot(list_2, kde = True, ax = axs[1], hist = True, bins = 100)
    
    axs[1].set_title('val Thereshold Curve')

def plot_decision_tree_and_importance(pipe, X_train):
    best_tree = pipe.best_estimator_.named_steps['estimator']
    
    # Normalize and plot feature importances
    feature_importances = best_tree.feature_importances_
    normalized_importances = feature_importances / np.max(feature_importances)
    
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': normalized_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette="viridis")
    plt.title("Feature Importances", fontsize=14)
    plt.xlabel("Importancia", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

def impact(target,predicted_probabilities):
    temp = pd.DataFrame(target).rename({'chargeback_dispute_status':'target'},axis=1)
    predicted_probabilities = [float(format(round(x,2),'.2f')) for x in predicted_probabilities]
    temp['score'] = predicted_probabilities
    temp['score'] = temp['score'].astype(float)
    # temp['score_1000'] = temp['score']*1000
    
    temp2 = temp.groupby('score').count().reset_index()
    temp2.sort_values(by='score',ascending=False,inplace=True)
    temp2.rename({'target':'volumetria'},axis=1,inplace=True)
    temp2['volume_acc_1'] = temp2['volumetria'].cumsum()
    
    temp3 = temp.groupby('score')['target'].sum().reset_index().sort_values(by='score',ascending=False)
    temp3.rename({'target':'acertos'},axis=1,inplace=True)
    temp3['acertos_acc'] = temp3['acertos'].cumsum()
    temp_final = temp2.merge(temp3, on='score', how='left')
    temp_final['erros'] = temp_final['volumetria'] - temp_final['acertos'] 
    temp_final['erros_acc'] = temp_final['volume_acc_1'] - temp_final['acertos_acc']
    temp_final.sort_values(by='score',ascending=False,inplace=True)
    
    prec_linha = []
    recall_linha = []
    for value in temp_final['score'].values:
        temp['y_predict'] = np.where(temp['score']>=value,1,0)
        prec_linha.append(precision_score(temp['target'],temp['y_predict']))
        recall_linha.append(recall_score(temp['target'],temp['y_predict']))
    temp_final['prec_linha'] = prec_linha
    temp_final['recall_linha'] = recall_linha
    
    # Base para o impacto

    scores = [round(0.01 + i * 0.01, 2) for i in range(100)]
    df_impact = pd.DataFrame({'score': scores})
    df_impact['volumetria_10'] = 256
    last_row_value = 256 + len(df_impact) * 0.23
    df_impact.loc[df_impact['score'] == 1.0, 'volumetria_10'] = last_row_value
    df_impact['volume_acc_10'] = df_impact['volumetria_10'].cumsum()
    df_impact.rename({'score':'impacto_percent'},axis=1,inplace=True)
    
    closest_scores = []
    closest_volus = []
    closest_volu_accs = []
    closest_acertos = []
    closest_acertos_accs = []
    closest_erros = []
    closest_erros_accs = []
    closest_prec_linhas = []
    closest_recall_linhas = []    
    
    for _, row_bar in df_impact.iterrows():
        min_diff = float('inf')  # Initialize minimum difference to infinity
        closest_score = None

        # Loop through each row in foo
        for _, row_foo in temp_final.iterrows():
            diff = abs(row_bar['volume_acc_10'] - row_foo['volume_acc_1'])
            if diff < min_diff:
                min_diff = diff
                closest_score = row_foo['score']
                closest_volu = row_foo['volumetria']
                closest_volu_acc = row_foo['volume_acc_1']
                closest_acerto = row_foo['acertos']
                closest_acertos_acc = row_foo['acertos_acc']
                closest_erro = row_foo['erros']
                closest_erros_acc = row_foo['erros_acc']
                closest_prec_linha = row_foo['prec_linha']
                closest_recall_linha = row_foo['recall_linha']

        closest_scores.append(closest_score)
        closest_volus.append(closest_volu)
        closest_volu_accs.append(closest_volu_acc)
        closest_acertos.append(closest_acerto)
        closest_acertos_accs.append(closest_acertos_acc)
        closest_erros.append(closest_erro)
        closest_erros_accs.append(closest_erros_acc)
        closest_prec_linhas.append(closest_prec_linha)
        closest_recall_linhas.append(closest_recall_linha)

    # Add the closest scores to the bar dataframe
    df_impact['closest_score'] = closest_scores
    df_impact['closest_volu'] = closest_volus
    df_impact['closest_volu_acc'] = closest_volu_accs
    df_impact['closest_acertos'] = closest_acertos
    df_impact['closest_acertos_acc'] = closest_acertos_accs
    df_impact['closest_erros'] = closest_erros
    df_impact['closest_erros_acc'] = closest_erros_accs
    df_impact['closest_prec_linha'] = closest_prec_linhas
    df_impact['closest_recall_linha'] = closest_recall_linhas
    
    return df_impact

def impact_grouped(target,predicted_probabilities, X_set, y_set):
    temp = pd.DataFrame(target).rename({'chargeback_dispute_status':'target'},axis=1)
    predicted_probabilities = [float(format(round(x,2),'.2f')) for x in predicted_probabilities]
    temp['score'] = predicted_probabilities
    temp['score'] = temp['score'].astype(float)
    temp.sort_values(by='score',ascending=False,inplace=True)

    temp2 = temp.groupby('score').count()
    temp2 = temp2.sort_values(by='score',ascending=False)
    temp2['volumetria_acc'] = temp2['target'].cumsum()
    temp2 = temp2.reset_index()
    temp2.rename({'target':'volumetria'},axis=1,inplace=True)
    
    
    temp3 = temp.groupby('score')['target'].sum().reset_index().sort_values(by='score',ascending=False)
    temp3.rename({'target':'acertos'},axis=1,inplace=True)
    temp3['acertos_acc'] = temp3['acertos'].cumsum()
    temp_final = temp2.merge(temp3, on='score', how='left')
    temp_final['erros'] = temp_final['volumetria'] - temp_final['acertos'] 
    temp_final['erros_acc'] = temp_final['volumetria_acc'] - temp_final['acertos_acc']
    temp_final.sort_values(by='score',ascending=False,inplace=True)


    prec_linha = []
    recall_linha = []
    for value in temp_final['score'].values:
        temp['y_predict'] = np.where(temp['score']>=value,1,0)
        prec_linha.append(precision_score(temp['target'],temp['y_predict']))
        recall_linha.append(recall_score(temp['target'],temp['y_predict']))
    temp_final['prec_linha'] = prec_linha
    temp_final['recall_linha'] = recall_linha

    bins = [min(temp_final['score']) - 0.01] + [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    bins.sort()
    # labels = [f'{bins[i]:.2f}' for i in range(1, len(bins))] 
    # Entrega o valor como 1.00, 0.95, 0.90
    labels = [f'({bins[i]:.2f},{bins[i+1]:.2f}]' for i in range(len(bins) - 1)] 
    # Entrega como intervalo aberto a esquerda fechado a direita 0.95,1.00

    temp_final['score_interval'] = pd.cut(temp_final['score'], bins=bins, right=True, labels=labels)

    grouped = temp_final.groupby('score_interval').agg({
        'volumetria': 'sum',
        'acertos': 'sum',
        'volumetria_acc': 'max',
        'acertos_acc': 'max',
        'erros': 'sum',
        'erros_acc': 'max',
        'prec_linha': 'mean',
        'recall_linha': 'mean'
    }).reset_index().sort_values(by='score_interval',ascending=False)
    
    grouped['percent_impact'] = grouped.apply(lambda x: (x['volumetria_acc']/(X_set.shape[0]))*100,axis=1)
    grouped['percent_error'] = grouped.apply(lambda x: (x['erros_acc']/(y_set.value_counts()[0]))*100,axis=1) 
    grouped['percent_acertos'] = grouped.apply(lambda x: (x['acertos_acc']/(y_set.value_counts()[1]))*100,axis=1)
    
    grouped['prec_acc'] = grouped.apply(lambda x: x['acertos_acc']/x['volumetria_acc'] ,axis=1)
    grouped['recall_acc'] = grouped.apply(lambda x: x['acertos_acc']/(y_set.value_counts()[1]) ,axis=1)

    return grouped

def calculate_simple_gains(df: pd.DataFrame):
    df['assign'] = np.where(df['y_true'] == df['y_pred'],1,0)
    valor_recuperado = df[(df['y_true'] == 1) & (df['assign'] == 1)]['Valor'].sum()
    qtd_recuperado = df[(df['y_true'] == 1) & (df['assign'] == 1)]['Valor'].shape[0]
    valor_total = df[(df['y_true'] == 1) & (df['assign'].isin([0,1]))]['Valor'].sum()
    qtd_total = df[(df['y_true'] == 1) & (df['assign'].isin([0,1]))]['Valor'].shape[0]

    valor_erroneamente_predito = df[(df['y_true'] == 0) & (df['y_pred'] == 1)]['Valor'].sum()
    qtd_erroneamente_predito = df[(df['y_true'] == 0) & (df['y_pred'] == 1)]['Valor'].shape[0]

    print('####################################### RESULTADOS POSITIVOS #####################################')
    print(f'Valor que o modelo identificou: {valor_recuperado}')
    print(f'Rate do valor monetário que o modelo inteferiu: {round(valor_recuperado/valor_total,2)}')
    print(f'Qtd de casos inteferidos pelo modelo {qtd_recuperado}')
    print(f'Rate de casos que o modelo inteferiu do total {round(qtd_recuperado/qtd_total,2)}')
    print('##################################################################################################')
    print('')
    print('')
    print('####################################### RESULTDOS NEGATIVOS ######################################')
    print(f'Valor que o modelo determinou erroneamente {valor_erroneamente_predito}')
    print(f'Qtd de casos em que o modelo determinou erroneamente {qtd_erroneamente_predito}')
    print('##################################################################################################')