# Projet Machine Learning : Prédiction du Risque de Maladie Cardiaque

## 📋 Description du projet

Ce projet de Machine Learning vise à prédire le risque de maladie cardiaque chez les patients en utilisant des caractéristiques médicales simples. Il s'agit d'un problème de classification binaire où le modèle doit déterminer si un patient présente un risque de maladie cardiaque (target = 1) ou non (target = 0).

Le projet suit le processus CRISP-DM (Cross-Industry Standard Process for Data Mining) :
1. **Business Understanding** : Compréhension du contexte médical et des objectifs
2. **Data Understanding** : Analyse exploratoire des données
3. **Data Visualization** : Visualisation des données et des relations
4. **Data Preparation** : Nettoyage, préparation et normalisation des données
5. **Modeling** : Entraînement de plusieurs modèles de classification
6. **Evaluation** : Comparaison des performances des modèles
7. **Deployment** : Déploiement via une application web Flask

## 🎯 Objectifs

- Analyser un dataset médical contenant des caractéristiques de patients
- Identifier les variables les plus corrélées au risque cardiaque
- Construire et comparer plusieurs modèles de Machine Learning
- Sélectionner le meilleur modèle selon le F1-score
- Déployer le modèle via une application web interactive

## 📁 Structure du projet

```
projet-ML/
├── app/
│   ├── app.py              # Application Flask principale
│   ├── model.pkl           # Modèle entraîné sauvegardé
│   ├── scaler.pkl          # Scaler pour normalisation
│   └── templates/
│       └── index.html      # Interface web
├── data/
│   └── heart.csv           # Dataset des patients
├── notebook/
│   └── projet_ml_heart_disease.ipynb  # Notebook d'analyse
├── presentation/
│   └── plan_presentation.md           # Plan de soutenance
├── train_model.py          # Script d'entraînement alternatif
├── README.md               # Ce fichier
└── questions_oral.md        # Questions pour la validation orale
```

## 🛠️ Technologies utilisées

### Data Science & Machine Learning
- **Python 3.14** : Langage de programmation principal
- **Pandas** : Manipulation et analyse des données
- **NumPy** : Calculs numériques
- **Scikit-learn** : Modèles de Machine Learning et métriques
  - LogisticRegression
  - RandomForestClassifier
  - KNeighborsClassifier
  - SVC (Support Vector Machine)
  - StandardScaler
  - train_test_split
  - Métriques : accuracy, precision, recall, F1-score

### Visualisation
- **Matplotlib** : Graphiques de base
- **Seaborn** : Visualisations statistiques avancées

### Déploiement Web
- **Flask** : Framework web pour l'application
- **Joblib** : Sauvegarde et chargement des modèles

## 📊 Dataset

Le dataset `heart.csv` contient 1025 patients et 14 variables :

| Variable | Description | Type |
|----------|-------------|------|
| age | Âge du patient (années) | Numérique |
| sex | Sexe (0 = Femme, 1 = Homme) | Catégorielle |
| cp | Type de douleur thoracique (0-3) | Catégorielle |
| trestbps | Pression artérielle au repos (mm Hg) | Numérique |
| chol | Taux de cholestérol (mg/dl) | Numérique |
| fbs | Taux de sucre à jeun (0 = ≤120, 1 = >120) | Catégorielle |
| restecg | ECG au repos (0-2) | Catégorielle |
| thalach | Fréquence cardiaque maximale | Numérique |
| exang | Angine induite par l'exercice (0 = Non, 1 = Oui) | Catégorielle |
| oldpeak | Dépression ST induite par l'exercice | Numérique |
| slope | Pente du segment ST (0-2) | Catégorielle |
| ca | Nombre de vaisseaux colorés (0-3) | Catégorielle |
| thal | Test de thallium (0-3) | Catégorielle |
| target | Variable cible (0 = Pas de maladie, 1 = Maladie) | Cible |

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étape 1 : Cloner le repository
```bash
git clone <url-du-repository>
cd projet-ML
```

Ou accéder directement au projet :
```powershell
cd C:\Users\moham\Desktop\projet-ML
```

### Étape 2 : Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
# Sur Windows
venv\Scripts\activate
# Sur Linux/Mac
source venv/bin/activate
```

### Étape 3 : Installer les dépendances
```powershell
pip install -r requirements.txt
```

Ou installer les packages manuellement :
```powershell
pip install pandas numpy matplotlib seaborn scikit-learn flask joblib
```

## 💻 Utilisation

### Option 1 : Via le Notebook Jupyter

1. Lancer Jupyter Notebook :
```bash
jupyter notebook
```

2. Ouvrir `notebook/projet_ml_heart_disease.ipynb`

3. Exécuter toutes les cellules dans l'ordre (Kernel → Restart & Run All)

4. À la fin, les fichiers `model.pkl` et `scaler.pkl` seront générés dans le dossier `app/`

**Note :** La partie Evaluation du notebook contient :
- Le tableau comparatif des performances des modèles
- Les métriques détaillées (accuracy, precision, recall, F1-score)
- Une visualisation comparative des performances
- La matrice de confusion du meilleur modèle
- La conclusion du choix du modèle basée sur le F1-score

### Option 2 : Via le script d'entraînement

1. Exécuter le script :
```powershell
python train_model.py
```

2. Le script entraînera les 4 modèles, les comparera et sauvegardera le meilleur modèle

### Option 3 : Lancer l'application Flask

1. Assurez-vous que les fichiers `model.pkl` et `scaler.pkl` existent dans `app/`

2. Lancer l'application :
```powershell
cd app
python app.py
```

3. Ouvrir un navigateur et aller à : `http://127.0.0.1:5000`

4. Remplir le formulaire avec les données du patient et cliquer sur "Prédire"

## 📈 Résultats

### Modèles comparés

| Modèle | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.8033 | 0.8000 | 0.8485 | 0.8235 |
| Random Forest | 0.7541 | 0.7647 | 0.7879 | 0.7761 |
| KNN | 0.7869 | 0.7778 | 0.8485 | 0.8116 |
| SVM | 0.7705 | 0.7714 | 0.8182 | 0.7941 |

### Meilleur modèle

**Modèle sélectionné :** Logistic Regression

**Performance :**
- Accuracy : 0.8033
- Precision : 0.8000
- Recall : 0.8485
- F1-score : 0.8235

Le modèle sélectionné est celui avec le meilleur **F1-score**, car cette métrique offre un équilibre optimal entre precision et recall, ce qui est crucial dans un contexte médical pour minimiser à la fois les faux positifs et les faux négatifs.

## 🔧 Configuration

### Modification du port Flask

Pour changer le port de l'application Flask, modifier la dernière ligne de `app/app.py` :
```python
if __name__ == "__main__":
    app.run(debug=True, port=8080)  # Changer le port ici
```

### Modification des hyperparamètres

Les hyperparamètres des modèles peuvent être modifiés dans le notebook (cellule 78) :
```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}
```

## 📝 Notes importantes

### Nettoyage des données
- Le dataset contient des doublons (723 détectés)
- Les doublons sont supprimés lors de la préparation des données
- Aucune valeur manquante n'est présente dans le dataset

### Normalisation
- Les données sont normalisées avec StandardScaler (moyenne = 0, écart-type = 1)
- Le scaler est sauvegardé pour être réutilisé dans l'application Flask
- La normalisation est essentielle pour les modèles sensibles à l'échelle (Logistic Regression, KNN, SVM)

### Division Train/Test
- 80% des données pour l'entraînement
- 20% des données pour le test
- Stratification pour conserver la proportion des classes

## 🎓 Contexte académique

Ce projet a été réalisé dans le cadre du module **Machine Learning Appliqué**. Il illustre l'application complète du processus de Data Science, de l'analyse exploratoire des données au déploiement d'un modèle en production.

## 📝 Exemple de test

Voici un exemple de valeurs à tester dans le formulaire :

| Variable | Valeur |
|----------|-------|
| age | 52 |
| sex | 1 |
| cp | 0 |
| trestbps | 125 |
| chol | 212 |
| fbs | 0 |
| restecg | 1 |
| thalach | 168 |
| exang | 0 |
| oldpeak | 1.0 |
| slope | 2 |
| ca | 2 |
| thal | 3 |

## ⚠️ Limites du projet

- **Taille du dataset** : Après suppression des doublons, seulement 302 patients, ce qui est relativement petit
- **Hyperparamètres** : Non optimisés, utilisation des valeurs par défaut
- **Validation** : Pas de validation croisée, seulement un simple split train/test
- **Feature engineering** : Aucune création de nouvelles features
- **Interprétabilité** : Pas d'analyse de l'importance des features (feature importance, SHAP values)

## 🚀 Améliorations possibles

- **Optimisation des hyperparamètres** : GridSearchCV ou RandomizedSearchCV
- **Validation croisée** : k-fold cross-validation pour une évaluation plus robuste
- **Feature engineering** : Création de nouvelles features (ex: BMI, catégories d'âge)
- **Autres modèles** : XGBoost, LightGBM, réseaux de neurones
- **Interprétabilité** : SHAP values ou LIME pour expliquer les prédictions
- **Plus de données** : Augmenter la taille du dataset si possible
- **Déploiement cloud** : Déployer l'application sur un service cloud (AWS, Heroku, etc.)

## 🤝 Contribution

Ce projet est à but éducatif. Pour toute suggestion ou amélioration, n'hésitez pas à ouvrir une issue ou un pull request.

## 📄 Licence

Ce projet est fourni à des fins éducatives.

## 👨‍💻 Auteur

Projet réalisé dans le cadre du module Machine Learning Appliqué - Juin 2026

## 🔗 Ressources

- [Dataset Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [CRISP-DM Methodology](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=methodology-crisp-dm-help-business-understand-data-mining)
