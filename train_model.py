import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

print("Dataset chargé avec succès.")
print("Dimensions :", df.shape)

# Suppression des doublons
df = df.drop_duplicates()

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

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Évaluation rapide
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPerformance du modèle Random Forest :")
print("Accuracy :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall :", round(recall, 4))
print("F1-score :", round(f1, 4))

# Création du dossier app si nécessaire
os.makedirs(APP_DIR, exist_ok=True)

# Sauvegarde
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nFichiers sauvegardés avec succès :")
print("model.pkl :", MODEL_PATH)
print("scaler.pkl :", SCALER_PATH)

print("\nVérification :")
print("model.pkl existe :", os.path.exists(MODEL_PATH))
print("scaler.pkl existe :", os.path.exists(SCALER_PATH))
print("Contenu du dossier app :", os.listdir(APP_DIR))