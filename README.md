# journey-ml

## Introdução

### Contexto

Dentro do site do BuscaVest, temos o plano de implementar um módulo de web analytics, permitindo que consigamos analizar a potencializar nosso serviço. Este é uma demonstração do impacto desse módulo, vastamente simplificando sua operação.

### Objetivos

Queremos responder a 3 perguntas utilizando dados de navegação e machine learning:

1. Quais comportamentos determinam se um usuário vai criar uma conta ou não?
2. Quais comportamentos dizem se ele está mais propenso a gastar dinheiro com uma assinatura dentro do nosso site?
3. Como podemos sugerir o pacote de estudos correto para cada usuário?

## Workflow

1. Gere dados usando `generators/web_navigation.py` e `generators/profiles.py`
2. Rode a pipeline de ETL com os arquivos em `preprocessing` e receba resultados de performance no console.
3. Carregue o modelo gerado e teste contra dados gerados posteriormente em `results.ipynb`.

### Mockup de dados

Em um cenário completo, os dados de navegação e perfis viriam de um data lake e banco de dados. No escopo dessa atividade, sintetizamos datasets utilizando scripts em Python, seguindo heurísticas esperadas do comportamento de navageção. Estes scripts ficam no diretório `generators` e depositam em `datasets`. Os dados são randômicos, então pequenas divergências nos resultados são esperadas.

### Processamento de dados

Criamos três pipelines para o processamento e treinamento. Arquivos `preprocessing.py` acessam os dados crus e os deixam prontos para o treinamento dos modelos. Foram aplicadas técnicas como Label, MinMax e OneHot Encoding demonstradas em sala de aula. Nessa etapa foram usadas as bibliotecas pandas, matplotlib, yellowbrick e seaborn. Houve o trabalho de descorir e refinar correlações e projetar dados para um formato acessível ao treinamento.

### Modelagem

### Aplicação do modelo

## Integrantes

| Jorge Terence | Rodrigo Zanetti | Vicente Venancio |
| --- | --- | --- |
| ![pfp jorge](https://avatars.githubusercontent.com/u/79718398?s=64&v=4) | ![pfp rodrigo](https://avatars.githubusercontent.com/u/116608232?s=64&v=4) | ![pfp vicente](https://avatars.githubusercontent.com/u/108531836?s=64&v=4) |

Nossos agradecimentos ao professor [Thiago Inocêncio](https://github.com/ThiagoInocencio), por compartilhar sua experiência.

## Melhorias futuras

1. Integrar com data lake e banco de dados oficial do projeto para buscar dados
2. Utilizar Spark para recuperar os dados e orquestrar processamentos utilizando Airflow
3. Criar demo de envio de emails e mensagens de marketing conectado com a interface administrativa
