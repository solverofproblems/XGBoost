from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
  
# fetch dataset 
breast_cancer = fetch_ucirepo(id=14) 
  
# data (as pandas dataframes) 
X = breast_cancer.data.features 
Y = breast_cancer.data.targets 
  
# metadata 
print(breast_cancer.metadata) 
  
# variable information 
print(breast_cancer.variables) 


y_mapping = {"no-recurrence-events" : 0, "recurrence-events" : 1}
y_numeric = Y.iloc[:, 0].map(y_mapping)

for col in X.columns:
    X[col] = X[col].astype('category')

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y_numeric, test_size=0.2, stratify=y_numeric, random_state=42
)

count_negative = (Y_train == 0).sum()
count_positive = (Y_train == 1).sum()
escala = count_negative / count_positive

modelo = XGBClassifier(
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = escala,
    enable_categorical = True,
    tree_method = "hist",
    early_stopping_rounds = 50,
    random_state = 42,
    eval_metric = 'logloss'
)

modelo.fit(

    X_train, Y_train,
    eval_set=[(X_test, Y_test)],
    verbose=False 

)

preds = modelo.predict(X_test)
preds_proba = modelo.predict_proba(X_test)[:, 1]

# Calcular métricas
acuracia = accuracy_score(Y_test, preds)
precisao = precision_score(Y_test, preds)
recall = recall_score(Y_test, preds)
f1 = f1_score(Y_test, preds)
auc_roc = roc_auc_score(Y_test, preds_proba)

print("\n" + "="*50)
print("MÉTRICAS DE DESEMPENHO")
print("="*50)
print(f"Acurácia:  {acuracia:.4f}")
print(f"Precisão:  {precisao:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")
print("="*50)

print("\n--- Relatório de Classificação ---")
print(classification_report(Y_test, preds))

print("\n--- Matriz de Confusão ---")
print(confusion_matrix(Y_test, preds))

# Visualizar Matriz de Confusão
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de Confusão em Heatmap
cm = confusion_matrix(Y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0],
            xticklabels=['Sem Recorrência', 'Recorrência'],
            yticklabels=['Sem Recorrência', 'Recorrência'])
axes[0].set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Verdadeiro', fontsize=12)
axes[0].set_xlabel('Predito', fontsize=12)

# Curva ROC
fpr, tpr, _ = roc_curve(Y_test, preds_proba)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Taxa de Falsos Positivos', fontsize=12)
axes[1].set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
axes[1].set_title('Curva ROC', fontsize=14, fontweight='bold')
axes[1].legend(loc="lower right", fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()