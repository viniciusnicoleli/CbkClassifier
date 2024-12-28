import os, sys
sys.path.insert(0, os.path.abspath(".."))
from dotenv import load_dotenv ; load_dotenv()

from datetime import datetime
import pandas as pd
import numpy as np

class BuildFeatures:
    def __init__(self):
        self._open_data() # Carrega os dados
        self._features_standard() # Cria as features
        self._save_data() # Salva os dados

    def _open_data(self):
        self.path_start = os.getenv("PATH_START")
        self.df = pd.read_csv(f"{self.path_start}data\\silver_dados_stone.csv")
        self.df.drop(['Unnamed: 0'],axis=1, inplace=True)

    def _save_data(self):
        self.df.to_csv(f"{self.path_start}data\\gold_dados_stone.csv")

    def _features_standard(self):
        self.df['dia_hora'] = pd.to_datetime(self.df['Dia'] + ' ' + self.df['Hora'])
        self.df['hour'] = self.df.apply(lambda x: x['dia_hora'].hour,axis=1)
        self.df['shift'] = pd.cut(self.df['hour'], bins=[0, 7, 12, 18, 24], labels=["1", "2", "3", "4"], right=False)
        self.df['week_number_month'] = self.df.apply(lambda x: (x['dia_hora'].day -1) // 7 + 1, axis=1)

        self.df['weekday'] = self.df.apply(lambda x: x['dia_hora'].strftime("%A")[:3],axis=1)
        weekday_mapping = {'Mon': 1, 
                           'Tue': 2, 
                           'Wed': 3, 
                           'Thu': 4, 
                           'Fri': 5, 
                           'Sat': 6, 
                           'Sun': 7}
        self.df['weekday_num'] = self.df['weekday'].map(weekday_mapping)
        self.df['shift'] = self.df['shift'].astype(int)
        self.df['weekday_num'] = self.df['weekday_num'].astype(int)

        self.__features_perfil()

    def __features_perfil(self):
        self.df['mean_shift_customer_no_cbk'] = self.df.apply(lambda x: self.get_basic_statistics_customer(
                                                                                    id_query=x['Cartão'],
                                                                                    dia_hora=x['dia_hora'],
                                                                                    col='shift',
                                                                                    cbk='Não'),axis=1)

        self.df['mean_shift_customer_cbk'] = self.df.apply(lambda x: self.get_basic_statistics_customer(
                                                                                    id_query=x['Cartão'],
                                                                                    dia_hora=x['dia_hora'],
                                                                                    col='shift',
                                                                                    cbk='Sim'),axis=1)
        
        self.df['mean_weekday_num_customer_no_cbk'] = self.df.apply(lambda x: self.get_basic_statistics_customer(
                                                                                    id_query=x['Cartão'],
                                                                                    dia_hora=x['dia_hora'],
                                                                                    col='weekday_num',
                                                                                    cbk='Não'),axis=1)

        self.df['mean_weekday_num_customer_cbk'] = self.df.apply(lambda x: self.get_basic_statistics_customer(
                                                                                    id_query=x['Cartão'],
                                                                                    dia_hora=x['dia_hora'],
                                                                                    col='weekday_num',
                                                                                    cbk='Sim'),axis=1)
        
        self.df['mean_days_operations_customer_no_cbk'] = self.df.apply(lambda x: self.get_days_customer(
                                                                                    id_query=x['Cartão'],
                                                                                    dia_hora=x['dia_hora'],
                                                                                    cbk='Não'),axis=1)

        self.df['mean_days_operations_customer_cbk'] = self.df.apply(lambda x: self.get_days_customer(
                                                                                    id_query=x['Cartão'],
                                                                                    dia_hora=x['dia_hora'],
                                                                                    cbk='Sim'),axis=1)

        self.df['last_day_no_cbk'] = self.df.apply(lambda x: self.get_last_day(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    cbk='Não'),axis=1)

        self.df['last_day_cbk'] = self.df.apply(lambda x: self.get_last_day(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    cbk='Sim'),axis=1)
        
        self.df['last_two_minutes_no_cbk_ops'] = self.df.apply(lambda x: self.get_days_operations(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    cbk='Não',
                                                    time_seconds=120),axis=1)

        self.df['last_two_minutes_cbk_ops'] = self.df.apply(lambda x: self.get_days_operations(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    cbk='Sim',
                                                    time_seconds=120),axis=1)

        self.df['last_five_minutes_no_cbk_ops'] = self.df.apply(lambda x: self.get_days_operations(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    cbk='Não',
                                                    time_seconds=300),axis=1)

        self.df['last_five_minutes_cbk_ops'] = self.df.apply(lambda x: self.get_days_operations(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    cbk='Sim',
                                                    time_seconds=300),axis=1)   

        self.df['last_two_minutes_all_ops'] = self.df.apply(lambda x: self.get_days_all_operations(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    time_seconds=120),axis=1)

        self.df['last_five_minutes_all_ops'] = self.df.apply(lambda x: self.get_days_all_operations(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    time_seconds=300),axis=1)   
        
        self.df['last_eight_minutes_all_ops'] = self.df.apply(lambda x: self.get_days_all_operations(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    time_seconds=480),axis=1)         

        self.df['qtd_cbk_operations_15d'] = self.df.apply(lambda x: self.get_qtd_cbk_days(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    days=15),axis=1)

        self.df['qtd_all_operations_15d'] = self.df.apply(lambda x: self.get_qtd_ops(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    days=15),axis=1)

        self.df['mean_time_all_operations_15d'] = self.df.apply(lambda x: self.get_mean_time_ops_all(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora']),axis=1)

        self.df['mean_revenue_all_operations_15d'] = self.df.apply(lambda x: self.get_revenue_last_ops(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    days=15,
                                                    type='mean'),axis=1)

        self.df['sum_revenue_all_operations_15d'] = self.df.apply(lambda x: self.get_revenue_last_ops(
                                                    id_query=x['Cartão'],
                                                    dia_hora=x['dia_hora'],
                                                    days=15,
                                                    type='sum'),axis=1)  
        
        self.df['last_purchase_time_between'] = self.df.apply(lambda x: x['last_day_no_cbk'] if x['last_day_no_cbk'] <= x['last_day_cbk'] else x['last_day_cbk'], axis=1)


        # Variáveis adicionadas posteriormente
        self.df['is_first_payment'] = self.df.apply(lambda x: self.is_first_payment(
                                        id_query=x['Cartão'],
                                        dia_hora=x['dia_hora']),axis=1)
        
        self.df['last_purchase_in_seconds_all_ops'] = self.df.apply(lambda x: self.get_last_purchase_seconds(
                                        id_query=x['Cartão'],
                                        dia_hora=x['dia_hora']),axis=1)  

        self.df['is_last_payment_equal'] = self.df.apply(lambda x: self.is_last_payment_equal(
                                        id_query=x['Cartão'],
                                        dia_hora=x['dia_hora'],
                                        payment=x['Valor']),axis=1)      

        self.df['mean_shift_customer_no_cbk'].fillna(-1, inplace=True)
        self.df['mean_shift_customer_cbk'].fillna(-1, inplace=True)
        self.df['mean_weekday_num_customer_no_cbk'].fillna(-1, inplace=True)
        self.df['mean_weekday_num_customer_cbk'].fillna(-1, inplace=True)
        self.df['mean_revenue_all_operations_15d'].fillna(-1, inplace=True)
        self.df['last_purchase_time_between'].fillna(-1, inplace=True)

    def get_basic_statistics_customer(self, id_query: str, dia_hora: datetime, col: str, cbk):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora) & 
            (self.df['CBK'] == cbk)
        ]
        return df_check[col].mean()

    def get_days_customer(self, id_query: str, dia_hora: datetime, cbk):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora) & 
            (self.df['CBK'] == cbk)
        ]
        if not df_check.empty:
            df_check['diff_days'] = df_check.apply(lambda x: (dia_hora - x['dia_hora']).days, axis=1)
            return df_check['diff_days'].mean()
        else:
            return -1

    def get_last_day(self,id_query: str, dia_hora: datetime, cbk):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora) & 
            (self.df['CBK'] == cbk)
        ]
        if not df_check.empty:
            return (dia_hora - df_check['dia_hora'].max()).total_seconds() / 60
        
    def get_days_operations(self,id_query: str, dia_hora, cbk: str, time_seconds: int):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora) & 
            (self.df['CBK'] == cbk)
        ]
        if not df_check.empty:
            df_check['diff_time'] = df_check.apply(lambda x: (dia_hora - x['dia_hora']).total_seconds(), axis=1)
            return df_check[df_check['diff_time'] <= time_seconds].shape[0]
        else:
            return -1
        
    def get_days_all_operations(self,id_query: str, dia_hora, time_seconds: int):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora)
        ]
        if not df_check.empty:
            df_check['diff_time'] = df_check.apply(lambda x: (dia_hora - x['dia_hora']).total_seconds(), axis=1)
            return df_check[df_check['diff_time'] <= time_seconds].shape[0]
        else:
            return -1

    def get_qtd_cbk_days(self,id_query: str, dia_hora, days: int):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora) & 
            (self.df['CBK'] == "Sim")
        ]
        if not df_check.empty:
            df_check['diff_time'] = df_check.apply(lambda x: (dia_hora - x['dia_hora']).days, axis=1)
            return df_check[df_check['diff_time'] <= days].shape[0]
        else:
            return -1

    def get_qtd_ops(self,id_query: str, dia_hora, days: int):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora)
        ]
        if not df_check.empty:
            df_check['diff_time'] = df_check.apply(lambda x: (dia_hora - x['dia_hora']).days, axis=1)
            return df_check[df_check['diff_time'] <= days].shape[0]
        else:
            return -1

    def get_mean_time_ops_all(self,id_query: str, dia_hora):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora)
        ]
        if not df_check.empty:
            df_check['diff_days'] = df_check.apply(lambda x: (dia_hora - x['dia_hora']).days, axis=1)
            return df_check['diff_days'].mean()
        else:
            return -1
    
    def get_revenue_last_ops(self,id_query: str, dia_hora, days, type):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora)
        ]
        if not df_check.empty:
            df_check['diff_time'] = df_check.apply(lambda x: (dia_hora - x['dia_hora']).days, axis=1)
            if type == 'sum':
                return df_check[df_check['diff_time'] <= days]['Valor'].sum()
            else:
                return df_check[df_check['diff_time'] <= days]['Valor'].mean()
        else:
            return -1

    def is_first_payment(self, id_query: str, dia_hora):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora)
        ]
        if df_check.empty:
            return 1
        else:
            return 0
        
    def get_last_purchase_seconds(self, id_query: str, dia_hora):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora)
        ]
        if not df_check.empty:
            return (dia_hora - df_check['dia_hora'].max()).total_seconds()
        else:
            return -1
        
    def is_last_payment_equal(self, id_query: str, dia_hora, payment):
        df_check = self.df[
            (self.df['Cartão'] == id_query) & 
            (self.df['dia_hora'] < dia_hora)
        ]
        if not df_check.empty:
            valor_ultima_compra = df_check.sort_values(by='dia_hora').iloc[-1]['Valor']
            return 1 if valor_ultima_compra == payment else 0
        else:
            return -1
        
if __name__ == '__main__':
    BuildFeatures()