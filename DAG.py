from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bigquery_analysis_and_results',
    default_args=default_args,
    description='Query do BigQuery processada com Python',
    schedule_interval='0 7 * * 4',  # Toda quinta as 7h
    catchup=False,
)


def fetch_data_from_bigquery():
    # Funcao simulada para pegar os dados do BigQuery
    # e organiza-los em um DataFrame
    pass


def process_transactions(**kwargs):
    ti = kwargs['ti']
    sams = ti.xcom_pull(task_ids='fetch_data_from_bigquery')

    # com sams sendo o DataFrame
    transactions = sams.groupby('ticket')['item_id'].apply(list).tolist()

    encoder = TransactionEncoder()
    encoded_transactions = encoder.fit_transform(transactions)
    df_transactions = pd.DataFrame(encoded_transactions, columns=encoder.columns_)

    frequent_itemsets = apriori(df_transactions, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    X = 1215794  # Item de Interesse
    rules['antecedents'] = rules['antecedents'].apply(lambda x: set(map(str, x)))
    filtered_rules = rules[rules['antecedents'] == {str(X)}]
    filtered_rules_sorted = filtered_rules.sort_values(by='confidence', ascending=False)


    print(filtered_rules_sorted)


t1 = BigQueryOperator(
    task_id='fetch_data_from_bigquery',
    sql='Q1.sql',  # ou `sql=Q1_query` se a query for armazenada em uma variavel
    use_legacy_sql=False,
    params={'date_max': '{{ macros.ds_add(ds, 0) }}', 'date_min': '{{ macros.ds_add(ds, -3) }}'},
    dag=dag,
)

t2 = PythonOperator(
    task_id='process_transactions',
    python_callable=process_transactions,
    provide_context=True,
    dag=dag,
)
#ordem de execucao
t1 >> t2
