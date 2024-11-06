# Análise de Investimentos em Mídia para Previsão de Vendas

Este projeto foi feito como desafio da formação de Cientista de Dados da escola DNC e realiza uma análise de investimentos em diferentes mídias (Youtube, Facebook e Jornal) para prever vendas com um modelo de regressão linear. A análise inclui a exploração de dados, visualizações e treinamento de um modelo preditivo. Este repositório foi desenvolvido como parte do meu portfólio e utiliza Python e diversas bibliotecas de ciência de dados.

## Tabela de Conteúdos

1. [Introdução](#introdução)
2. [Objetivo](#objetivo)
3. [Instalação](#instalação)
4. [Descrição do Código](#descrição-do-código)
   - [1. Importação de Bibliotecas](#1-importação-de-bibliotecas)
   - [2. Carregamento e Exploração de Dados](#2-carregamento-e-exploração-de-dados)
   - [3. Análise Exploratória de Dados (EDA)](#3-análise-exploratória-de-dados-eda)
   - [4. Remoção de Outliers](#4-remoção-de-outliers)
   - [5. Preparação dos Dados para Treinamento](#5-preparação-dos-dados-para-treinamento)
   - [6. Treinamento do Modelo de Regressão Linear](#6-treinamento-do-modelo-de-regressão-linear)
   - [7. Avaliação do Modelo](#7-avaliação-do-modelo)
   - [8. Interpretação dos Coeficientes](#8-interpretação-dos-coeficientes)
5. [Conclusão](#conclusão)

## Introdução

Este projeto aplica técnicas de regressão linear para prever vendas com base em diferentes tipos de investimentos em mídia (Youtube, Facebook e Jornal). O objetivo é entender quais plataformas geram o maior retorno sobre o investimento.

## Objetivo

- Prever as vendas com base nos investimentos em diferentes mídias.
- Analisar quais canais de mídia (Youtube, Facebook ou Jornal) têm maior impacto nas vendas.

## Instalação

Para executar este projeto, você precisa das seguintes bibliotecas:

```bash
pip install pandas seaborn matplotlib scikit-learn
```

## Descrição do Código

### 1. Importação de Bibliotecas

O código começa importando as bibliotecas essenciais para o projeto:
- `pandas` para manipulação de dados.
- `seaborn` e `matplotlib` para visualização de dados.
- `sklearn` para modelagem e avaliação do modelo de regressão linear.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

### 2. Carregamento e Exploração de Dados

O dataset é carregado a partir de um arquivo CSV chamado `MKT.csv`. Em seguida, são exibidos os primeiros registros e informações gerais sobre os dados, incluindo estatísticas descritivas e a verificação de valores nulos.

```python
df = pd.read_csv('MKT.csv')
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

### 3. Análise Exploratória de Dados (EDA)

Esta seção inclui:
- **Distribuição dos Investimentos:** Histogramas são criados para visualizar a distribuição dos investimentos em `youtube`, `facebook` e `newspaper`.
- **Matriz de Correlação:** A matriz de correlação e seu heatmap visualizam as relações entre as variáveis.
- **Scatter Plots:** A função `pairplot` do Seaborn cria scatter plots entre variáveis para explorar relações potenciais.

```python
# Exemplo de código para visualização
sns.pairplot(df)
plt.show()
```

### 4. Remoção de Outliers

Uma função personalizada é definida para remover outliers usando o método IQR (Interquartile Range), aplicando-a nas colunas `youtube`, `facebook`, `newspaper` e `sales`. 

```python
def remove_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]
```

### 5. Preparação dos Dados para Treinamento

As variáveis independentes (investimentos em `youtube`, `facebook`, e `newspaper`) e a variável dependente (`sales`) são definidas. O dataset é dividido em conjuntos de treino e teste, com 20% dos dados sendo reservados para teste.

```python
x = df[['youtube','facebook','newspaper']]
y = df['sales']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### 6. Treinamento do Modelo de Regressão Linear

O modelo de regressão linear é instanciado e treinado com os dados de treino.

```python
modelo = LinearRegression()
modelo.fit(x_train, y_train)
```

### 7. Avaliação do Modelo

Após o treinamento, o modelo é avaliado com as seguintes métricas:
- **MSE (Erro Quadrático Médio):** Mede o erro médio ao quadrado.
- **R² (Coeficiente de Determinação):** Mede a proporção da variância dos dados explicada pelo modelo.

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}")
print(f"R²: {r2}")
```

### 8. Interpretação dos Coeficientes

Os coeficientes do modelo indicam o impacto de cada variável independente (`youtube`, `facebook`, `newspaper`) nas vendas:
- **Youtube:** Coeficiente de 0.044186 indica que cada unidade adicional de investimento em `youtube` aumenta as vendas em 0.044 unidades.
- **Facebook:** Coeficiente de 0.194482, o que representa um impacto positivo ligeiramente maior nas vendas em comparação ao Youtube.
- **Newspaper:** Coeficiente de -0.000049 indica um impacto levemente negativo nas vendas, sugerindo que o investimento em `newspaper` pode não ser eficiente.

```python
coeficientes = pd.DataFrame({
    'Feature': x.columns,
    'Coeficiente': modelo.coef_
})
print(coeficientes)
```

## Conclusão

Este projeto demonstra o uso de regressão linear para prever vendas com base em investimentos em diferentes plataformas de mídia. Os resultados indicam que:

- **Facebook** tem o maior impacto positivo entre as variáveis analisadas, enquanto o **Youtube** tem um impacto menor, mas ainda positivo.
- O **Newspaper** apresenta um impacto negativo insignificante, sugerindo que este canal de mídia não traz retorno significativo.

Essa análise pode orientar decisões sobre onde concentrar os investimentos em mídia para maximizar as vendas. Em futuros estudos, outras variáveis e modelos mais complexos poderiam ser explorados para melhorar a precisão e relevância das previsões.
