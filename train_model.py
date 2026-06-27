import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
APP_DIR = os.path.join(BASE_DIR, "app")

MODEL_PATH = os.path.join(APP_DIR, "model.pkl")
SCALER_PATH = os.path.join(APP_DIR, "scaler.pkl")

# Vérification du dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

# Chargement du dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Entraînement des modèles - Prédiction du risque de maladie cardiaque")
print("=" * 60)
print("\nDataset chargé avec succès.")
print("Dimensions :", df.shape)

# Suppression des doublons
df = df.drop_duplicates()
print(f"Après suppression des doublons : {df.shape}")

# Séparation X / y
X = df.drop("target", axis=1)
y = df["target"]

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nDivision train/test :")
print(f"Train : {X_train.shape}")
print(f"Test : {X_test.shape}")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nNormalisation effectuée avec StandardScaler.")

# Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=42)
}

# Entraînement et évaluation
results = []
trained_models = {}

print("\n" + "=" * 60)
print("Entraînement des modèles")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })
    
    print(f"\n{name} entraîné avec succès.")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")

# Création du tableau comparatif
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-score", ascending=False)

print("\n" + "=" * 60)
print("Tableau comparatif des modèles")
print("=" * 60)
print(results_df.to_string(index=False))

# Sélection du meilleur modèle
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print("Meilleur modèle sélectionné")
print("=" * 60)
print(f"Modèle : {best_model_name}")
print(f"F1-score : {results_df.iloc[0]['F1-score']:.4f}")

# Création du dossier app si nécessaire
os.makedirs(APP_DIR, exist_ok=True)

# Sauvegarde
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n" + "=" * 60)
print("Sauvegarde des fichiers")
print("=" * 60)
print(f"Modèle sauvegardé : {MODEL_PATH}")
print(f"Scaler sauvegardé : {SCALER_PATH}")

print("\nVérification :")
print(f"model.pkl existe : {os.path.exists(MODEL_PATH)}")
print(f"scaler.pkl existe : {os.path.exists(SCALER_PATH)}")
print(f"Contenu du dossier app : {os.listdir(APP_DIR)}")

print("\n" + "=" * 60)
print("Entraînement terminé avec succès !")
print("=" * 60)