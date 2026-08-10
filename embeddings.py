import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import sys

# ---------------------------- Feature Calculation Functions ----------------------------
def compute_pca_features(data: np.ndarray, K: int = 3) -> np.ndarray:
    """
    Calcula os top K autovalores da matriz de covariância dos dados.
    """
    if data.shape[0] < 2 or data.shape[1] == 0:
        return np.zeros(K)
    
    try:
        cov = np.cov(data, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        return eigvals[:K]
    except Exception as e:
        print(f"compute_pca_features error: {e}")
        return np.zeros(K)


def differential_entropy_multivariate(data: np.ndarray) -> float:
    """
    Calcula a entropia diferencial de uma normal multivariada com base na matriz de covariância.
    Fórmula: H = 0.5 * ln((2πe)^d * det(Σ))
    Implementado usando slogdet para estabilidade numérica.
    """
    n, d = data.shape
    if n < 2 or d == 0:
        return 0.0
    
    try:
        cov = np.cov(data, rowvar=False)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0 or not np.isfinite(logdet):
            return 0.0
        # H = 0.5 * (d * ln(2πe) + ln(det(cov)))
        entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
        return entropy
    except Exception as e:
        print(f"differential_entropy_multivariate error: {e}")
        return 0.0


def singular_values_cross_covariance(data: np.ndarray, top_k: int = 3) -> np.ndarray:
    """
    Calcula os top_k valores singulares da matriz de cross-covariância entre passado e futuro da série.
    Centraliza os dados antes de calcular.
    """
    n, d = data.shape
    if n < 2:
        return np.zeros(top_k)

    try:
        X_past = data[:-1]
        X_future = data[1:]
        # Centralizar colunas
        X_past = X_past - X_past.mean(axis=0)
        X_future = X_future - X_future.mean(axis=0)
        cross_cov = np.dot(X_past.T, X_future) / (n - 1)
        _, s, _ = np.linalg.svd(cross_cov)
        return s[:top_k]
    except Exception as e:
        print(f"singular_values_cross_covariance error: {e}")
        return np.zeros(top_k)


def compute_var_features(data: np.ndarray, p: int = 1) -> np.ndarray:
    """
    Ajusta um VAR(p) multivariado e retorna os coeficientes achatados A_k, normalizando Y e verificando colinearidade.
    """
    N, d = data.shape
    if N <= p or d == 0:
        return np.zeros(d * d * p)

    try:
        # Construir matrizes de defasagem
        Y_list = []
        for lag in range(1, p + 1):
            Y_list.append(data[:-lag])  # X_{t-lag}
        Y = np.hstack(Y_list)
        X = data[p:]

        # Normalizar Y
        means = np.mean(Y, axis=0)
        stds = np.std(Y, axis=0)
        stds[stds == 0] = 1.0
        Y_norm = (Y - means) / stds
        # Verificar colinearidade
        cond = np.linalg.cond(Y_norm)
        # if cond > 1e10:
        #     print(f"Warning: high condition number in VAR design matrix: {cond}")

        # Resolver A pelo método de mínimos quadrados
        A_mat, *_ = np.linalg.lstsq(Y_norm, X, rcond=None)
        return A_mat.flatten()
    except np.linalg.LinAlgError as e:
        print(f"compute_var_features LinAlgError: {e}")
        return np.zeros(d * d * p)
    except Exception as e:
        print(f"compute_var_features error: {e}")
        return np.zeros(d * d * p)


def compute_residual_features(data: np.ndarray, p: int = 1, M: int = 3) -> np.ndarray:
    """
    Ajusta um VAR(p), calcula a covariância dos resíduos e retorna
    os top M autovalores dessa covariância, normalizando Y.
    """
    N, d = data.shape
    if N <= p or d == 0:
        return np.zeros(M)

    try:
        # Construir Y e X para VAR(p)
        Y_list = [data[:-lag] for lag in range(1, p + 1)]
        Y = np.hstack(Y_list)
        X = data[p:]

        # Normalizar Y
        means = np.mean(Y, axis=0)
        stds = np.std(Y, axis=0)
        stds[stds == 0] = 1.0
        Y_norm = (Y - means) / stds
        # Verificar colinearidade
        cond = np.linalg.cond(Y_norm)
        #if cond > 1e10:
            #print(f"Warning: high condition number in residual VAR design matrix: {cond}")

        # Ajustar VAR
        A_mat, *_ = np.linalg.lstsq(Y_norm, X, rcond=None)
        # Calcular resíduos
        resid = X - Y_norm.dot(A_mat)
        res_cov = np.cov(resid, rowvar=False)
        # Extrair autovalores
        eigvals = np.linalg.eigvalsh(res_cov)
        eigvals = np.sort(eigvals)[::-1]
        return eigvals[:M]
    except np.linalg.LinAlgError as e:
        print(f"compute_residual_features LinAlgError: {e}")
        return np.zeros(M)
    except Exception as e:
        print(f"compute_residual_features error: {e}")
        return np.zeros(M)

# ---------------------------- Original Functions ----------------------------
def calculate_basic_stats(window_df):
    """
    Calcula estatísticas básicas dos fluxos na janela.
    """
    stats = {}
    
    # Estatísticas de duração dos fluxos
    stats['flow_duration_mean'] = window_df['FLOW_DURATION_MILLISECONDS'].mean()
    stats['flow_duration_stddev'] = window_df['FLOW_DURATION_MILLISECONDS'].std()
    stats['flow_duration_min'] = window_df['FLOW_DURATION_MILLISECONDS'].min()
    stats['flow_duration_max'] = window_df['FLOW_DURATION_MILLISECONDS'].max()
    
    # Estatísticas de tamanho dos fluxos
    stats['flow_size_mean'] = window_df['FLOW_SIZE'].mean()
    stats['flow_size_stddev'] = window_df['FLOW_SIZE'].std()
    stats['flow_size_min'] = window_df['FLOW_SIZE'].min()
    stats['flow_size_max'] = window_df['FLOW_SIZE'].max()
    
    # Estatísticas de IAT
    stats['src_to_dst_iat_mean'] = window_df['SRC_TO_DST_IAT_AVG'].mean()
    stats['src_to_dst_iat_stddev'] = window_df['SRC_TO_DST_IAT_AVG'].std()
    stats['src_to_dst_iat_min'] = window_df['SRC_TO_DST_IAT_AVG'].min()
    stats['src_to_dst_iat_max'] = window_df['SRC_TO_DST_IAT_AVG'].max()
    stats['dst_to_src_iat_mean'] = window_df['DST_TO_SRC_IAT_AVG'].mean()
    stats['dst_to_src_iat_stddev'] = window_df['DST_TO_SRC_IAT_AVG'].std()
    stats['dst_to_src_iat_min'] = window_df['DST_TO_SRC_IAT_AVG'].min()
    stats['dst_to_src_iat_max'] = window_df['DST_TO_SRC_IAT_AVG'].max()
    
    # Contagem de fluxos
    stats['num_flows'] = len(window_df)
    
    return stats

def prepare_data_for_advanced_features(window_df):
    """
    Prepara os dados da janela para as funções avançadas.
    Normaliza o tempo e seleciona as features principais.
    Retorna matriz com 5 colunas: [normalized_time, flow_size, src_to_dst_iat, dst_to_src_iat, flow_duration]
    """
    if len(window_df) < 2:
        return np.array([]).reshape(0, 5)
    
    # Selecionar as colunas principais
    time_col = window_df['FLOW_START_MILLISECONDS'].values
    flow_size_col = window_df['FLOW_SIZE'].values
    src_to_dst_iat_col = window_df['SRC_TO_DST_IAT_AVG'].values
    dst_to_src_iat_col = window_df['DST_TO_SRC_IAT_AVG'].values
    flow_duration_col = window_df['FLOW_DURATION_MILLISECONDS'].values
    
    # Normalizar o tempo (0 a 1 dentro da janela)
    if time_col.max() > time_col.min():
        normalized_time = (time_col - time_col.min()) / (time_col.max() - time_col.min())
    else:
        normalized_time = np.zeros_like(time_col)
    
    # Combinar em matriz
    data = np.column_stack([
        normalized_time, 
        flow_size_col, 
        src_to_dst_iat_col, 
        dst_to_src_iat_col, 
        flow_duration_col
    ])
    
    # Remover linhas com NaN ou infinitos
    data = data[np.isfinite(data).all(axis=1)]
    
    return data

def calculate_advanced_features(window_df):
    """
    Calcula características avançadas dos fluxos na janela usando as novas funções.
    """
    features = {}
    
    # Preparar dados
    data = prepare_data_for_advanced_features(window_df)
    
    if data.shape[0] < 2:
        # Retornar features zeradas se não há dados suficientes
        features.update({f'pca_eigenval_{i+1}': 0.0 for i in range(3)})
        features['differential_entropy'] = 0.0
        features.update({f'cross_cov_singular_{i+1}': 0.0 for i in range(3)})
        return features
    
    # PCA features
    pca_vals = compute_pca_features(data, K=3)
    for i, val in enumerate(pca_vals):
        features[f'pca_eigenval_{i+1}'] = val
    
    # Entropia diferencial
    features['differential_entropy'] = differential_entropy_multivariate(data)
    
    # Cross-covariance singular values
    cross_cov_vals = singular_values_cross_covariance(data, top_k=3)
    for i, val in enumerate(cross_cov_vals):
        features[f'cross_cov_singular_{i+1}'] = val
    
    return features

def calculate_temporal_features(window_df):
    """
    Calcula características temporais dos fluxos na janela usando VAR.
    """
    features = {}
    
    # Preparar dados
    data = prepare_data_for_advanced_features(window_df)
    
    if data.shape[0] < 3:  # VAR precisa de pelo menos 3 pontos
        # Retornar features zeradas
        d = 3  # dimensões dos dados
        p = 1  # lag
        features.update({f'var_coeff_{i+1}': 0.0 for i in range(d * d * p)})
        features.update({f'var_residual_eigenval_{i+1}': 0.0 for i in range(3)})
        return features
    
    # VAR features
    var_coeffs = compute_var_features(data, p=1)
    for i, coeff in enumerate(var_coeffs):
        features[f'var_coeff_{i+1}'] = coeff
    
    # Residual features
    residual_vals = compute_residual_features(data, p=1, M=3)
    for i, val in enumerate(residual_vals):
        features[f'var_residual_eigenval_{i+1}'] = val
    
    return features

def create_embeddings(window_df):
    """
    Cria embeddings da janela chamando todas as funções auxiliares de cálculo.
    
    Args:
        window_df: DataFrame com os fluxos da janela
    
    Returns:
        Dict com todas as características calculadas
    """
    embeddings = {}
    
    # Calcular estatísticas básicas
    basic_stats = calculate_basic_stats(window_df)
    embeddings.update(basic_stats)
    
    # Calcular características avançadas
    advanced_features = calculate_advanced_features(window_df)
    embeddings.update(advanced_features)
    
    # Calcular características temporais
    temporal_features = calculate_temporal_features(window_df)
    embeddings.update(temporal_features)
    
    if all(val == 0 for val in embeddings.values()):
        print("Warning: embedding nulo criado para esta janela")

    
    return embeddings

def create_time_windows(df, window_size_ms=500):
    """
    Divide os fluxos em janelas de tempo fixas e gera embeddings para cada janela.
    
    Args:
        df: DataFrame com os fluxos
        window_size_ms: Tamanho da janela em milissegundos
    
    Returns:
        DataFrame com embeddings de cada janela
    """
    print(f"Iniciando janeamento temporal com janelas de {window_size_ms}ms")
    
    # Remover coluna Label se existir
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
    
    # Ordenar por timestamp
    df_sorted = df.sort_values('FLOW_START_MILLISECONDS').reset_index(drop=True)
    
    # Encontrar o timestamp mínimo e máximo
    min_time = df_sorted['FLOW_START_MILLISECONDS'].min()
    max_time = df_sorted['FLOW_START_MILLISECONDS'].max()
    
    print(f"Período dos dados: {min_time} a {max_time} ms")
    print(f"Duração total: {(max_time - min_time) / 1000:.2f} segundos")
    
    # Calcular o número de janelas necessárias
    total_duration = max_time - min_time
    num_windows = int(np.ceil(total_duration / window_size_ms))
    
    print(f"Número total de janelas: {num_windows}")
    
    # Lista para armazenar os embeddings e tempos
    embeddings_list = []
    embedding_times = []
    
    # Criar as janelas
    for i in tqdm(range(num_windows), desc="Processando janelas"):
        # Definir o início e fim da janela atual
        window_start = min_time + (i * window_size_ms)
        window_end = window_start + window_size_ms
        
        # Filtrar fluxos que começam nesta janela
        window_flows = df_sorted[
            (df_sorted['FLOW_START_MILLISECONDS'] >= window_start) & 
            (df_sorted['FLOW_START_MILLISECONDS'] < window_end)
        ].copy()
        
        # Só processar janelas que tenham pelo menos um fluxo
        if len(window_flows) > 0:
            # Medir tempo de criação do embedding
            start_time = time.time()
            
            # Criar embeddings da janela
            window_embeddings = create_embeddings(window_flows)
            
            # Calcular tempo decorrido
            end_time = time.time()
            embedding_time = end_time - start_time
            embedding_times.append(embedding_time)
            
            # Adicionar metadados da janela
            window_embeddings['embedding_creation_time_seconds'] = embedding_time
            
            embeddings_list.append(window_embeddings)
    
    # Converter para DataFrame
    embeddings_df = pd.DataFrame(embeddings_list)
    
    # Calcular e imprimir estatísticas de tempo
    if embedding_times:
        mean_time = np.mean(embedding_times)
        min_time_emb = np.min(embedding_times)
        max_time_emb = np.max(embedding_times)
        total_time = np.sum(embedding_times)
        
        print(f"\nEstatísticas de tempo dos embeddings:")
        print(f"Tempo médio por embedding: {mean_time:.6f} segundos")
        print(f"Tempo mínimo: {min_time_emb:.6f} segundos")
        print(f"Tempo máximo: {max_time_emb:.6f} segundos")
        print(f"Tempo total: {total_time:.6f} segundos")
    
    print(f"Embeddings gerados: {len(embeddings_df)}")
    print(f"Características por embedding: {len(embeddings_df.columns)}")
    
    return embeddings_df

class TeeOutput:
    """Classe para redirecionar saída tanto para terminal quanto para arquivo"""
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()

def load_and_window_flows(file_path, window_size_ms=500, output_file=None):
    """
    Carrega o arquivo de fluxos, divide em janelas temporais e salva os embeddings.
    
    Args:
        file_path: Caminho para o arquivo com os fluxos
        window_size_ms: Tamanho da janela em milissegundos
        output_file: Nome do arquivo de saída (opcional)
    
    Returns:
        DataFrame com os embeddings
    """
    print(f"Carregando dados de: {file_path}")
    
    # Carregar os dados
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    
    print(f"Dados carregados: {len(df)} fluxos")
    print(f"Colunas disponíveis: {list(df.columns)}")
    
    # Criar embeddings das janelas temporais
    embeddings_df = create_time_windows(df, window_size_ms)
    
    # Definir nome do arquivo de saída
    if output_file is None:
        base_name = file_path.split('.')[0]
        output_file = f"{base_name}_embeddings_{window_size_ms}ms.parquet"
    
    # Salvar embeddings
    embeddings_df.to_parquet(output_file, index=False)
    print(f"Embeddings salvos em: {output_file}")
    
    return embeddings_df

if __name__ == '__main__':
    entradas = ["netflow_v3/CICIDS/benign_netflow_CICIDS.parquet"]
    saidas = ["benign_CICIDS.txt"]
    
    for entrada, saida in zip(entradas, saidas):
    
        # Configurar redirecionamento da saída para arquivo
        output_log = open(saida, "w")
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(sys.stdout, output_log)
        
        try:
            # Configurações
            WINDOW_SIZE_MS = 1000  # 500 milissegundos
            
            # Arquivo de entrada
            input_file = entrada
            
            print(f"Iniciando processamento - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            start_total = time.time()
            
            # Processar os dados e salvar embeddings
            embeddings_df = load_and_window_flows(input_file, WINDOW_SIZE_MS)
            
            end_total = time.time()
            total_processing_time = end_total - start_total
            
            print(f"\nProcessamento concluído!")
            print(f"Tempo total de processamento: {total_processing_time:.2f} segundos")
            print(f"Total de embeddings gerados: {len(embeddings_df)}")
            print(f"Características por embedding: {len(embeddings_df.columns)}")
            
            print(f"\nFinalizado em: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        finally:
            # Restaurar saída padrão e fechar arquivo
            sys.stdout = original_stdout
            output_log.close()
            print("Log salvo em: output.txt")
