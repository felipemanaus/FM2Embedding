# FM2Embedding: Time Series Embeddings of Network Flows for Semi-Supervised Anomaly Detection

**Artigo Relacionado:**
* **Título:** Time Series Embeddings of Network Flows for Semi-Supervised Anomaly Detection
* **Autores:** Fernando Nakayama, Felipe Melo, Gabriel Violante, Michele Nogueira (UFMG / UFPR / UFRRJ)
* **Referência:** ISCC 2026

---

## Descrição do Projeto

Este repositório contém a implementação do método **FM2Embedding** (Flow Metadata to Embedding), proposto para a detecção de anomalias de rede utilizando aprendizado semi-supervisionado. O método captura a dinâmica temporal e as propriedades estatísticas dos fluxos de rede, reduzindo drasticamente o volume de dados (representando cerca de 5.39% da base original) e superando a necessidade de pipelines complexos de pré-processamento.

Os fluxos de rede são organizados em janelas de tempo fixas e transformados em séries temporais multivariadas. A partir dessas janelas, o método extrai recursos estruturais e temporais (Estatísticas Básicas, Autovalores PCA, Entropia Diferencial multivariada, Cross-covariância e coeficientes de modelos autorregressivos - VAR), criando embeddings que alimentam algoritmos de Machine Learning não supervisionados/semi-supervisionados (como K-Means, Isolation Forest e Autoencoders).

---

## Estrutura do Repositório

O projeto é construído em dois eixos principais: a geração de embeddings matemáticos temporais e os classificadores de detecção de anomalia.

### 1. Geração Avançada de Embeddings (`embeddings.py`)
Script responsável pela transformação dos fluxos brutos em embeddings compactos:
* Criação de janelas temporais de tamanho fixo (ex: 500ms, 1000ms, 3000ms, 5000ms).
* Normalização das variáveis dentro de cada janela.
* **Cálculos e Características (Features) implementadas:**
  * Estatísticas básicas de IAT e Tamanho dos fluxos.
  * **PCA:** Autovalores da matriz de covariância.
  * **Entropia Diferencial Multivariada:** Baseado na matriz de covariância.
  * **Valores Singulares da Cross-Covariância:** Avalia a dependência entre instantes passados e futuros.
  * **VAR (Vector Autoregression):** Coeficientes ajustados de inter-relação entre lags temporais.
  * **Autovalores Residuais:** Captura ruído ou variabilidade não linear baseada no VAR.

### 2. Modelos de Machine Learning para Detecção
* **K-Means (`kmeans_supervisionado.py`):** Utiliza agrupamento (clustering) para identificar anomalias sem o conhecimento prévio de rótulos (labels). Mede desempenho utilizando métricas como Ajusted Rand Index (ARI) e NMI.
* **Isolation Forest (`isolationforest_benign.py`):** Treina exclusivamente em instâncias benignas (ou majoritariamente benignas, assumindo uma contaminação). Excepcional para tratar a heterogeneidade das anomalias.

*(O artigo também cita o uso de Autoencoders que superaram o desempenho, cuja implementação deve seguir o processo de reconstrução do espaço latente dos embeddings gerados).*

---

## Pré-requisitos e Instalação

Python 3.8+ é recomendado. Instale as dependências:

```bash
pip install pandas numpy tqdm scikit-learn
```
(Para processamento de arquivos parquet, garanta que a biblioteca `pyarrow` ou `fastparquet` esteja instalada).

---

## Como Executar (Guia de Reprodução)

### Passo 1: Geração dos Embeddings (Feature Extraction)
Você deve rodar o gerador de embeddings para transformar seu Dataset em janelas com representações numéricas condensadas.

1. Baixe o dataset de rede em formato parquet (Ex: `benign_netflow_CICIDS.parquet`).
2. Edite as variáveis no final do script de geração (como `WINDOW_SIZE_MS = 1000`) para o tamanho de janela desejado (o artigo recomenda testar 1s, 3s e 5s).
3. Rode o gerador:
   ```bash
   python embeddings.py
   ```
4. O processo em lote (tqdm progress) gerará os arquivos `.parquet` contendo os embeddings e um log do tempo gasto.

### Passo 2: Treinar e Testar - Isolation Forest
Este modelo é treinado apenas com fluxo benigno (ou assumindo leve contaminação) e avalia anomalias.

1. No arquivo respectivo, configure os caminhos do dataset gerado no Passo 1 (`train_path` e `test_path`).
2. Execute o modelo:
   ```bash
   python isolationforest_benign.py
   ```
3. O script calculará *AUC-ROC, F1-Score, Detection Rate e Matriz de Confusão*.

### Passo 3: Treinar e Testar - K-Means
Executa a clusterização não supervisionada das janelas de tempo.

1. Verifique os caminhos dos embeddings no arquivo do K-means.
2. Defina os hiperparâmetros (como `k=8` ou `k=2` de acordo com a janela escolhida conforme o artigo).
3. Execute o script para extrair as métricas (FPR, TPR, ARI, NMI):
   ```bash
   python kmeans_supervisionado.py
   ```

---

## Resultados e Contribuições do Artigo
A utilização do FM2Embedding resultou em:
* Redução agressiva de dados computados e processados (apenas cerca de 2% a 5% de retenção em relação aos fluxos raw).
* **Treinamento e inferência muito mais rápidos** comparado à métodos tradicionais (reduções de tempo superiores a 90%).
* Com modelos semi-supervisionados e usando amostras não etiquetadas, alcançou-se valores de **F1-Score até 0.84** (Isolation Forest) e até **0.89 ARI** com K-Means. Modelos profundos (Autoencoders) atingiram **F1-Score de até 0.95**.

---
