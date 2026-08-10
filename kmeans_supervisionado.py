import pandas as pd
import numpy as np
import time  # ADICIONADO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import roc_auc_score


def calculate_detection_metrics(y_true, y_pred_clusters, attack_label='attack', threshold=0.5):
    """
    Calcula Detection Rate e False Positive Rate para detecÃ§Ã£o de ataques,
    permitindo mÃºltiplos clusters de ataque e mÃ©tricas de similaridade de clusters.

    y_true: labels verdadeiros ('attack' ou 'benign')
    y_pred_clusters: clusters preditos (0, 1, 2, ...)
    attack_label: qual label representa ataque
    threshold: proporÃ§Ã£o mÃ­nima de ataques em um cluster para considerÃ¡-lo de ataque

    Retorna: dict com mÃ©tricas
    """
    # Converte para binÃ¡rio (1=attack, 0=benign)
    y_true_binary = (y_true == attack_label).astype(int)

    unique_clusters = np.unique(y_pred_clusters)
    cluster_counts = {}
    cluster_attack_counts = {}
    for cluster in unique_clusters:
        mask = (y_pred_clusters == cluster)
        cluster_counts[cluster] = mask.sum()
        cluster_attack_counts[cluster] = np.sum(y_true_binary[mask])

    # Identifica clusters de ataque pelo threshold
    attack_clusters = [c for c in unique_clusters
                       if cluster_attack_counts[c] / cluster_counts[c] >= threshold]

    # Mapeia prediÃ§Ãµes binÃ¡rias: 1 se cluster em attack_clusters, 0 caso contrÃ¡rio
    y_pred_binary = np.isin(y_pred_clusters, attack_clusters).astype(int)

    # Calcula matriz de confusÃ£o
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    # Calcula mÃ©tricas de detecÃ§Ã£o
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * detection_rate) / (precision + detection_rate) if (precision + detection_rate) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    auc = roc_auc_score(y_true_binary, y_pred_binary)

    # Gera relatÃ³rio de classificaÃ§Ã£o
    report = classification_report(y_true_binary, y_pred_binary, target_names=['benign', 'attack'], output_dict=True)

    # MÃ©tricas de similaridade de cluster vs labels
    ari = adjusted_rand_score(y_true, y_pred_clusters)
    nmi = normalized_mutual_info_score(y_true, y_pred_clusters, average_method='arithmetic')

    metrics = {
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'precision': precision,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'attack_clusters': attack_clusters,
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
        'classification_report': report,
        'ARI': ari,
        'NMI': nmi,
        'AUC': auc
    }

    return metrics



def calculate_clustering_metrics(X, labels, y_true=None):
    """
    Calcula mÃ©tricas de clustering
    """
    metrics = {}
    
    # MÃ©tricas internas (sem labels verdadeiros)
    metrics['silhouette_score'] = silhouette_score(X, labels)
    metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    
    # MÃ©tricas externas (com labels verdadeiros)
    if y_true is not None:
        # Converte y_true para numÃ©rico se necessÃ¡rio
        if isinstance(y_true.iloc[0], str):
            y_true_numeric = (y_true == 'attack').astype(int)
        else:
            y_true_numeric = y_true
            
        metrics['adjusted_rand_score'] = adjusted_rand_score(y_true_numeric, labels)
    
    return metrics

def train_kmeans(train_file, k, numeric_columns, scale_data, label_column):
    """
    Treina o modelo K-means nos dados de treino
    
    train_file: arquivo parquet de treino
    k: nÃºmero de clusters
    numeric_columns: colunas para usar (None = todas numÃ©ricas)
    scale_data: se deve normalizar os dados
    label_column: nome da coluna com labels ('attack'/'benign')
    
    Retorna: DataFrame treino com clusters, modelo KMeans, scaler, mÃ©tricas, tempo_treino
    """
    # Carrega dados de treino
    df_train = pd.read_parquet(train_file)
    print(f"Dados de treino carregados: {len(df_train)} linhas")
    
    # Seleciona colunas numÃ©ricas (exclui coluna de label se existir)
    if numeric_columns is None:
        numeric_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if label_column and label_column in numeric_columns:
            numeric_columns.remove(label_column)
    
    print(f"Colunas utilizadas: {numeric_columns}")
    
    # Prepara dados de treino
    X_train = df_train[numeric_columns].copy()
    
    # Preenche NaN com mÃ©dia
    if X_train.isnull().any().any():
        print("Preenchendo valores NaN com a mÃ©dia...")
        X_train = X_train.fillna(X_train.mean())
    
    df_train_clean = df_train.copy()
    
    # Normaliza dados se solicitado
    scaler = None
    if scale_data:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print("Dados de treino normalizados")
    else:
        X_train_scaled = X_train.values
    
    # MEDE TEMPO DE TREINO - ADICIONADO
    start_time = time.time()
    
    # Treina K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    train_labels = kmeans.fit_predict(X_train_scaled)
    
    # FINALIZA MEDIÇÃO DE TEMPO - ADICIONADO
    training_time = time.time() - start_time
    
    # Adiciona clusters ao DataFrame de treino
    df_train_clean['cluster'] = train_labels
    
    # EstatÃ­sticas do treino
    print(f"\n=== TREINO ===")
    print(f"Linhas processadas: {len(df_train_clean)}")
    print(f"InÃ©rcia (WCSS): {kmeans.inertia_:.2f}")
    print(f"Tempo de treino: {training_time:.4f} segundos")  # ADICIONADO
    print(f"DistribuiÃ§Ã£o de clusters:")
    print(df_train_clean['cluster'].value_counts().sort_index())
    
    # Calcula mÃ©tricas de detecÃ§Ã£o se hÃ¡ labels
    detection_metrics = None
    if label_column and label_column in df_train_clean.columns:
        detection_metrics = calculate_detection_metrics(
            df_train_clean[label_column], 
            df_train_clean['cluster']
        )
        
        print(f"\n=== MÃ‰TRICAS DE DETECÃ‡ÃƒO (TREINO) ===")
        print(f"Detection Rate (TPR): {detection_metrics['detection_rate']:.4f}")
        print(f"False Positive Rate: {detection_metrics['false_positive_rate']:.4f}")
        print(f"Precision: {detection_metrics['precision']:.4f}")
        print(f"F1-Score: {detection_metrics['f1_score']:.4f}")
        print(f"Accuracy: {detection_metrics['accuracy']:.4f}")
        print(f"Clusters de Ataque: {detection_metrics['attack_clusters']}")
        print(f"ARI: {detection_metrics['ARI']}")
        print(f"NMI: {detection_metrics['NMI']}")
        print(f"AUC: {detection_metrics['AUC']}")
        
        cm = detection_metrics['confusion_matrix']
        print(f"Matriz de ConfusÃ£o: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    
    return df_train_clean, kmeans, scaler, numeric_columns, detection_metrics, training_time  # MODIFICADO

def predict_test(test_file, kmeans_model, scaler, numeric_columns, label_column=None):
    """
    Aplica o modelo treinado nos dados de teste
    
    test_file: arquivo parquet de teste
    kmeans_model: modelo treinado
    scaler: scaler ajustado no treino
    numeric_columns: colunas usadas no treino
    label_column: nome da coluna com labels
    
    Retorna: DataFrame teste com clusters preditos, mÃ©tricas
    """
    # Carrega dados de teste
    df_test = pd.read_parquet(test_file)
    print(f"\nDados de teste carregados: {len(df_test)} linhas")
    
    # Prepara dados de teste (mesmas colunas do treino)
    X_test = df_test[numeric_columns].copy()
    
    # Preenche NaN com mÃ©dia
    if X_test.isnull().any().any():
        print("Preenchendo valores NaN com a mÃ©dia...")
        X_test = X_test.fillna(X_test.mean())
    
    df_test_clean = df_test.copy()
    
    # Aplica mesma normalizaÃ§Ã£o do treino
    if scaler:
        X_test_scaled = scaler.transform(X_test)
        print("Dados de teste normalizados com scaler do treino")
    else:
        X_test_scaled = X_test.values
    
    # Prediz clusters
    test_labels = kmeans_model.predict(X_test_scaled)
    df_test_clean['cluster'] = test_labels
    
    # EstatÃ­sticas do teste
    print(f"\n=== TESTE ===")
    print(f"Linhas processadas: {len(df_test_clean)}")
    print(f"DistribuiÃ§Ã£o de clusters:")
    print(df_test_clean['cluster'].value_counts().sort_index())
    
    # Calcula mÃ©tricas de detecÃ§Ã£o se hÃ¡ labels
    detection_metrics = None
    if label_column and label_column in df_test_clean.columns:
        detection_metrics = calculate_detection_metrics(
            df_test_clean[label_column], 
            df_test_clean['cluster']
        )
        
        print(f"\n=== MÃ‰TRICAS DE DETECÃ‡ÃƒO (TESTE) ===")
        print(f"Detection Rate (TPR): {detection_metrics['detection_rate']:.4f}")
        print(f"False Positive Rate: {detection_metrics['false_positive_rate']:.4f}")
        print(f"Precision: {detection_metrics['precision']:.4f}")
        print(f"F1-Score: {detection_metrics['f1_score']:.4f}")
        print(f"Accuracy: {detection_metrics['accuracy']:.4f}")
        print(f"Clusters de Ataque: {detection_metrics['attack_clusters']}")
        print(f"ARI: {detection_metrics['ARI']}")
        print(f"NMI: {detection_metrics['NMI']}")
        print(f"AUC: {detection_metrics['AUC']}")
        
        cm = detection_metrics['confusion_matrix']
        print(f"Matriz de ConfusÃ£o: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    
    return df_test_clean, detection_metrics

def run_kmeans_train_test(train_file, test_file, k, numeric_columns=None, scale_data=True, label_column=None):
    """
    Executa treino e teste completo
    
    label_column: nome da coluna com labels ('attack'/'benign')
    
    Retorna: df_train, df_test, modelo, scaler, mÃ©tricas, tempo_treino
    """
    # Treina modelo - MODIFICADO
    df_train, kmeans_model, scaler, cols_used, train_detection_metrics, training_time = train_kmeans(
        train_file, k, numeric_columns, scale_data, label_column
    )
    
    # Aplica no teste
    df_test, test_detection_metrics = predict_test(
        test_file, kmeans_model, scaler, cols_used, label_column
    )
    
    # Organiza todas as mÃ©tricas
    all_metrics = {
        'train': {
            'detection': train_detection_metrics
        },
        'test': {
            'detection': test_detection_metrics
        },
        'training_time': training_time  # ADICIONADO
    }
    
    return df_train, df_test, kmeans_model, scaler, all_metrics

# Exemplo de uso
if __name__ == "__main__":
    dataset = "NB15" #CICIDS ou NB15
    interval = "original" #5000, 3000, 1000 ou original
    train_file = f"embeddings_final/{dataset}/{interval}/{dataset}_embs{interval}_treino70.parquet"
    test_file = f"embeddings_final/{dataset}/{interval}/{dataset}_embs{interval}_teste30.parquet"

    if interval == "original":
        label = "Attack"
    else:
        label = "Label"
        
    
    try:
        # Executa treino e teste
        df_train, df_test, model, scaler, metrics = run_kmeans_train_test(
            train_file=train_file,
            test_file=test_file,
            k=8,  #clusters para attack/benign
            scale_data=True,
            label_column=label  # substitua pelo nome da sua coluna de label
        )
        
        # Salva resultados
        # df_train.to_parquet("treino_com_clusters.parquet")
        # df_test.to_parquet("teste_com_clusters.parquet")
        
        print(f"\n=== RESULTADOS SALVOS ===")
        print("treino_com_clusters.parquet")
        print("teste_com_clusters.parquet")
        
        # Mostra centroides
        print(f"\n=== CENTROIDES ===")
        for i, centroid in enumerate(model.cluster_centers_):
            print(f"Cluster {i}: {centroid}")
        
        # Resumo das mÃ©tricas principais - MODIFICADO
        print(f"\n=== RESUMO DAS MÃ‰TRICAS ===")
        print(f"Tempo de treino: {metrics['training_time']:.4f} segundos")  # ADICIONADO
        
        if metrics['test']['detection']:
            test_det = metrics['test']['detection']
            print(f"TESTE - Detection Rate: {test_det['detection_rate']:.4f}")
            print(f"TESTE - False Positive Rate: {test_det['false_positive_rate']:.4f}")
            print(f"TESTE - F1-Score: {test_det['f1_score']:.4f}")
            print(f"TESTE - Accuracy: {test_det['accuracy']:.4f}")
            print(f"TESTE - ARI: {test_det['ARI']}")
            print(f"TESTE - NMI: {test_det['NMI']}")
            print(f"TESTE - AUC: {test_det['AUC']}")
        
    except FileNotFoundError as e:
        print(f"Arquivo nÃ£o encontrado: {e}")
    except Exception as e:
        print(f"Erro: {e}")