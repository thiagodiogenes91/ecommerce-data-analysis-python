#1. Bibliotecas: 

import pandas as pd #Para análise e manipução de Dados
import numpy as np #Para operações numéricas
import matplotlib.pyplot as plt # Para criação de gráficos básicos
import seaborn as sns # Para gráficos estatísticos

#2. Carregamento de Dados provenientes dos arquivos CVS:
#Leitura do arquivo CSV e armazenamento de dataframe

orders = pd.read_csv('olist_orders_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')
reviews = pd.read_csv('olist_order_reviews_dataset.csv')

#3. Integração de Dados:
#integração de dataframes usando chaves em comum
#Merge serve para unir tabelas

df = orders.merge(items, on='order_id') 
df = df.merge(products, on='product_id')
df = df.merge(customers, on='customer_id')
df = df.merge(reviews, on='order_id')

#4. Limpeza e tratamento dos dados:

# Aqui é feito uma conversão de colunas de data
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

# Aqui é feito remoção de valores nulos em colunas que são importantes para a análise
# dropna serve para limpar dados ao remover valores ausentes(Null).
df = df.dropna(subset=['order_delivered_customer_date']) #remove pedidos sem entrega
df = df.dropna(subset=['product_category_name']) #remove produtos sem categoria

# drop aqui remove colunas irrelevantes
df = df.drop(columns=['review_comment_title', 'review_comment_message'])

# Verifica a estrutura
print(df.info())

#5. Criação de novas variáveis:

# Valor total do pedido
df['total_value'] = df['price'] + df['freight_value']

# Tempo de entrega
df['delivery_time'] = (
    df['order_delivered_customer_date'] - df['order_purchase_timestamp']
).dt.days

# Remove valores inconsistentes (tempo negativos)
df = df[df['delivery_time'] >= 0]

#6. Análise Exploratória:

# Receita total por categoria de produto
revenue_category = df.groupby('product_category_name')['total_value'].sum().sort_values(ascending=False)
print("\nTop categorias por faturamento:")
print(revenue_category.head(10))

# Avaliação média por categoria
review_category = df.groupby('product_category_name')['review_score'].mean().sort_values()
print("\nCategorias com pior avaliação:")
print(review_category.head(10))

# Tempo médio de entrega por estado
delivery_state = df.groupby('customer_state')['delivery_time'].mean().sort_values(ascending=False)
print("\nEstados com maior tempo de entrega:")
print(delivery_state.head(10))

# Status dos pedidos
print("\nStatus dos pedidos:")
print(df['order_status'].value_counts())

# 7. Visualização de Dados (Gráficos)

# Top 10 categorias por faturamento
revenue_category.head(10).plot(kind='bar')
plt.title('Top 10 Categorias por Faturamento')
plt.xlabel('Categoria')
plt.ylabel('Receita')
plt.xticks(rotation=45)
plt.show()

# Avaliações
sns.countplot(x='review_score', data=df)
plt.title('Distribuição das Avaliações')
plt.show()

# Tempo médio de entrega por estado (top 10)
delivery_state.head(10).plot(kind='bar')
plt.title('Estados com Maior Tempo de Entrega')
plt.xlabel('Estado')
plt.ylabel('Dias')
plt.show()

#Distribuição do tempo de entrega
plt.hist(df['delivery_time'], bins=30)
plt.title('Distribuição do Tempo de Entrega')
plt.xlabel('Dias')
plt.ylabel('Quantidade de Pedidos')
plt.show()

#Faturamento por estado
df.groupby('customer_state')['total_value'].sum().sort_values(ascending=False).head(10).plot(kind='bar')

plt.title('Top 10 Estados por Faturamento')
plt.xlabel('Estado')
plt.ylabel('Receita')
plt.show()

#8. Função para achar os clientes inativos: 

def clientes_inativos(df, dias=90): #Defini o tempo de 90 dias
    ultima_data = df['order_purchase_timestamp'].max() #Data da última compra do cliente
    clientes = df.groupby('customer_unique_id')['order_purchase_timestamp'].max()
    
    inativos = clientes[(ultima_data - clientes).dt.days > dias]
    return inativos

print("\nClientes inativos:")
print(clientes_inativos(df).head())

print("Total de clientes inativos:", len(clientes_inativos(df)))

#Comparação Clientes Ativos x Clientes Inativos
todos_clientes = df['customer_unique_id'].nunique()

inativos = clientes_inativos(df)
qtd_inativos = len(inativos)

ativos = todos_clientes - qtd_inativos

print("Total de clientes:", todos_clientes)
print("Clientes inativos:", qtd_inativos)
print("Clientes ativos:", ativos)

print("Percentual inativos:", (qtd_inativos / todos_clientes) * 100)
print("Percentual ativos:", (ativos / todos_clientes) * 100)

#Gráfico de comparação:
labels = ['Ativos', 'Inativos']
valores = [ativos, qtd_inativos]

plt.bar(labels, valores)
plt.title('Clientes Ativos vs Inativos')
plt.show()


#9. Análise cruzada de valores totais por pedido.

# Cruzamento categoria x status
cross_analysis = df.groupby(['product_category_name', 'order_status'])['total_value'].sum()
print("\nAnálise cruzada (categoria x status):")
print(cross_analysis.head(20))