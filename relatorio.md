# SAM'S CLUB CASE

Resolução do case do Sam's Club para encontrar os itens mais relevantes e, a partir disso, ajudar o time comercial a montar combos promocionais. Também vamos utilizar um método de machine learning para decidir novos kits. Ao final, teremos uma pipeline com características específicas.
## IDE

`PyCharm` e `Jupyter`

## Instalação

Use o comando [pip](https://pip.pypa.io/en/stable/) para instalar as bibliotecas necessárias.  
Nesse caso usamos ``pandas`` e ``mlxtend``

```bash
pip install pandas mlxtend
```

## Parte 1 – Analytics  
```python
import pandas
import mlxtend

sams = pd.read_csv('C:PATH\CASE_PRATICO_SAMS_CLUB.csv')
sams
```
Identifiquei os 10 itens mais relevantes. E para cada item relevante, encontrei os 15 itens com maior frequencia na mesma compra. 
```python
grouped_sams = sams.groupby(['departamento','periodo', 'item_id', 'item_descricao'])
agg_sams = grouped_sams.agg({'receita_bruta': 'sum', 'item_unidade': 'sum', 'margem': 'sum'})
top10_agrupado = agg_sams.sort_values(by='margem', ascending=False).reset_index().head(30)
```

Também adicionei a coluna de ``Coeficiente`` para entender os itens mais relevantes. Essa coluna executa a seguinte operação  
```python  
marg_recbrut['Coeficiente'] = marg_recbrut['margem']/marg_recbrut['item_unidade'] 
```  
Ou seja, demonstra quais são os itens com maior lucro líquido por unidade.

Criei, então, o ``sams_simples_tratado`` com colunas adicionais ``Condicional`` e ``combo``  
A partir da escolha do valor ``X = item_id``, o ``df`` cria uma tabela com possíveis combos em ``['combo']`` com a seguinte operação:
```python
sams_simples_tratado['combo'] = item_unico + ' + ' + sams_simples_tratado['item_descricao']
```

Em seguida foi produzido um relatório One Page no Tableau que tem a possibilidade de cada comprador visualizar seu departamento e o gerente consiga ter uma visão macro do período.  

DataFrames preparados no Tableau Prep Builder 2023.3.



## Parte 2 – Data Science

Afim de entender melhor o que deveria ser feito, optei por criar a lógica inicial sem machine learning. Utilizando apenas as funcões do ``pandas``. 
Mantive `X = item_id` e adicionei `n_top_items` que determina a quantidade de itens para criação de um `custom_combo`.  

Com a lógica clara utilizei o `mlxtend`(machine leaning extension for python) e as funções `association_rules` e `apriori`. Ambas utilizadas para identificar possíveis `itemsets` e suas frequências.

`apriori`: utilizada para identificar conjunto de items frequentes. Através do `threshold` pude determinar a frequência desejada desses conjuntos dentro do `df` estudado.

`association_rules`: métricas que avaliam a probabilidade de ocorrência de determinados `itemsets`, utilizando as medidas de confiança(`confidence`) da informação, da dependência e independência(`lift`), da sociação e dissociação(`zhang_metric`) e de outras.

### Code

Para esse exercício adequei os dados em `sams` com a função `TransactionEncoder`da biblioteca `mlxtend`. 

Inicialmente transformei e agrupei os dados em listas:
```python 
  transactions = sams.groupby('ticket')['item_id'].apply(list).tolist()
```
#### Method: One-hot coding
Utilizei o `TransactionEnconder` para transformar a lista de transações em um formato de matriz `bool`
```python
encoder = TransactionEncoder()
encoded_transactions = encoder.fit_transform(transactions)
df_transactions = pd.DataFrame(encoded_transactions, columns=encoder.columns_)
```
#### Funcoes: `apriori` e `association_rules`
Utilizando a função `apriori`, configurei o suporte mínimo para transações com `min_support=0.01` e ativei a opção `use_colnames=True`, permitindo assim o uso dos nomes dos itens em vez de índices numéricos. 

Ao usar `association_rules`, adotei uma lógica similar à estrutura `if-then`, expressa como se antecedente, então consequente, utilizando a métrica de `confidence` com um `min_threshold=0.1`, o que equivale a uma confiança mínima de 10%.

Entao filtrei e setei o valor de `X` com o item `antecedent`(convertido para `string`) com o seguinte trecho: 
```python
X = 1215794  # O item de interesse
rules['antecedents'] = rules['antecedents'].apply(lambda x: set(map(str, x)))
filtered_rules = rules[rules['antecedents'] == {str(X)}]
filtered_rules_sorted = filtered_rules.sort_values(by='confidence', ascending=False)
```
O resultado retorna uma determinada lista de itens com todos os valores das métricas para serem interpretados. Neste caso tivemos o seguinte retorno:  
````python
  antecedents consequents  antecedent support  consequent support   support  \
4   {1215794}   (1236279)            0.077795            0.176444  0.016450   
0   {1215794}   (1215821)            0.077795            0.051195  0.012431   
2   {1215794}   (1215823)            0.077795            0.048608  0.011887   

   confidence      lift  leverage  conviction  zhangs_metric  
4    0.211460  1.198452  0.002724    1.044406       0.179559  
0    0.159789  3.121156  0.008448    1.129245       0.736935  
2    0.152797  3.143426  0.008105    1.122979       0.739397  

````
Podemos dizer que o item 1215823 é comprado 3.14 vezes(`lift`) mais em conjunto com o item 1215794 do que sozinho. E com `zhangs_metrics` podemos dizer que ambos os itens tem forte associação com o `antecedent` pois os valores estão mais próximos de 1.  
Então os dois últimos `consequents` são possíveis para um bom combo promocional. 

## Parte 3 – Pipeline de dados/Airflow:.

Primeiro rodei os imports necessários para definir a `DAG`
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
```
Setei os argumentos padrão
```python 
     default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}
```
Defini a `DAG` como `bigquery_analysis_and_results` com agendamento para às 7 horas nas Quintas. Sem recuperações para execuções passadas

```python
dag = DAG(
    'bigquery_analysis_and_results',
    default_args=default_args,
    description='Query BigQuery e processamento com Python para um Dashboard',
    schedule_interval='0 7 * * 4', 
    catchup=False,
) 
```
Criei a primeira tarefa `t1` que utiliza o `BigQueryOperator` para executar a busca no BigQuery. 

```python
t1 = BigQueryOperator(
    task_id='fetch_data_from_bigquery',
    sql='Q1.sql',
    use_legacy_sql=False,
    params={'date_max': '{{ macros.ds_add(ds, 0) }}', 'date_min': '{{ macros.ds_add(ds, -3) }}'},
    dag=dag,
)
```
Em seguida `t2` foi criada para executar a `process_transactions` do exercício 2. 
```python
t2 = PythonOperator(
    task_id='process_transactions',
    python_callable=process_transactions,
    provide_context=True,
    dag=dag,
)

```
Enfim criamos o encadeamento das tarefas:  
``t1 >> t2``