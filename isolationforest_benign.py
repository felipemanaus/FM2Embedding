from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import time


def treinar_testar_isolation_forest(X_train, y_train, X_test, y_test, contamination=0.1):
    """
    Treina e testa um modelo Isolation Forest para detecÃ§Ã£o de anomalias
    
    ParÃ¢metros:
    - X_train: features de treino
    - y_train: labels de treino ("attack", "benign")
    - X_test: features de teste
    - y_test: labels de teste ("attack", "benign")
    - contamination: proporÃ§Ã£o esperada de outliers (padrÃ£o: 0.1)
    """
    
    # Substitui NaNs pela mÃ©dia da coluna
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)
    
    # Converter labels para formato binÃ¡rio (1 para normal, -1 para anomalia)
    # "benign" = 1 (normal), "attack" = -1 (anomalia)
    y_train_binary = np.where(y_train == "benign", 1, -1)
    y_test_binary = np.where(y_test == "benign", 1, -1)
    
    # Para as mÃ©tricas finais, vamos usar 0 para benign e 1 para attack
    y_test_metrics = np.where(y_test == "attack", 1, 0)
    
    # Treinar apenas com dados normais (benign)
    X_train_normal = X_train[y_train == "benign"]
    
    print(f"\nTreinando com {len(X_train_normal)} amostras normais...")
    
    # Criar e treinar o modelo
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200
    )
    
    # Medir tempo de treino
    start_time = time.time()
    iso_forest.fit(X_train_normal)
    training_time = time.time() - start_time
    
    # Fazer prediÃ§Ãµes no conjunto de teste
    y_pred_binary = iso_forest.predict(X_test)
    
    # Converter prediÃ§Ãµes para formato 0/1 (0=benign, 1=attack)
    y_pred_metrics = np.where(y_pred_binary == -1, 1, 0)  # -1 (anomalia) -> 1 (attack)
    
    # Calcular scores de anomalia para AUC
    anomaly_scores = iso_forest.decision_function(X_test)
    
    # Calcular mÃ©tricas
    accuracy = accuracy_score(y_test_metrics, y_pred_metrics)
    precision = precision_score(y_test_metrics, y_pred_metrics, pos_label=1)  # 1 Ã© attack
    recall = recall_score(y_test_metrics, y_pred_metrics, pos_label=1)
    f1 = f1_score(y_test_metrics, y_pred_metrics, pos_label=1)
    
    # Detection Rate Ã© o mesmo que Recall para a classe de attack
    detection_rate = recall
    
    # AUC-ROC (scores mais negativos = mais anÃ´malos = mais provÃ¡vel de ser attack)
    auc = roc_auc_score(y_test_metrics, -anomaly_scores)
    
    # Matrix de confusÃ£o (0=benign, 1=attack)
    cm = confusion_matrix(y_test_metrics, y_pred_metrics, labels=[0, 1])
    
    # RelatÃ³rio de classificaÃ§Ã£o
    report = classification_report(y_test_metrics, y_pred_metrics, 
                                 target_names=['Benign', 'Attack'], 
                                 digits=2)
    
    # Imprimir resultados no formato solicitado
    print("\n=== Isolation Forest (sklearn) ===")
    print(f"AcurÃ¡cia       : {accuracy:.4f}")
    print(f"PrecisÃ£o       : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"Detection Rate : {detection_rate:.4f}")
    print(f"AUC (ROC)     : {auc:.4f}")
    print(f"Tempo Treino   : {training_time:.4f} segundos")
    
    print("\n--- Matriz de ConfusÃ£o (linhas = verdadeiro, colunas = predito) ---")
    print("               Pred: Benign(0)   Pred: Attack(1)")
    print(f"Verdadeiro 0 :     {cm[0,0]:5d}            {cm[0,1]:4d}")
    print(f"Verdadeiro 1 :      {cm[1,0]:4d}           {cm[1,1]:5d}")
    
    print("\n--- RelatÃ³rio por classe ---")
    print(report)
    
    return {
        'modelo': iso_forest,
        'acuracia': accuracy,
        'precisao': precision,
        'recall': recall,
        'f1_score': f1,
        'detection_rate': detection_rate,
        'auc_roc': auc,
        'confusion_matrix': cm,
        'tempo_treino': training_time
    }


# ConfiguraÃ§Ã£o dos datasets
dataset = "NB15" #CICIDS ou NB15
interval = "5000" #5000, 3000, 1000 ou original
train_path = f"embeddings_final/{dataset}/{interval}/{dataset}_embs{interval}_treino70.parquet"
test_path = f"embeddings_final/{dataset}/{interval}/{dataset}_embs{interval}_teste30.parquet"

if interval == "original":
    label = "Attack"
else:
    label = "Label"


label_col = label       # coluna de rÃ³tulo alternativa
positive_label = "attack"    # valor que representa ataque

# Carregar os dados
print("Carregando dados...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separar features e labels
X_train = train_df.drop(columns=[label_col])
y_train = train_df[label_col]
X_test = test_df.drop(columns=[label_col])
y_test = test_df[label_col]

print(f"Dados de treino: {X_train.shape}")
print(f"Dados de teste: {X_test.shape}")
print(f"DistribuiÃ§Ã£o treino: {y_train.value_counts()}")
print(f"DistribuiÃ§Ã£o teste: {y_test.value_counts()}")

resultados = treinar_testar_isolation_forest(X_train, y_train, X_test, y_test, contamination=0.15)