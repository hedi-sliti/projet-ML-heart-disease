# Plan de Présentation - Projet Machine Learning : Prédiction du Risque de Maladie Cardiaque

## Durée estimée : 10-15 minutes

---

## 1. Introduction (1-2 minutes)

### 1.1. Présentation du contexte
- **Contexte médical** : Les maladies cardiovasculaires sont l'une des principales causes de décès dans le monde
- **Importance de la détection précoce** : Permet de mettre en place des mesures préventives et d'améliorer les chances de survie
- **Problématique** : Comment prédire le risque de maladie cardiaque à partir de caractéristiques médicales simples ?

### 1.2. Présentation du projet
- **Objectif** : Construire un modèle de Machine Learning pour prédire le risque de maladie cardiaque
- **Type de problème** : Classification binaire (0 = pas de maladie, 1 = maladie)
- **Dataset** : 1025 patients avec 14 variables médicales
- **Approche méthodologique** : Suivi du processus CRISP-DM

---

## 2. Méthodologie (2-3 minutes)

### 2.1. Processus CRISP-DM
Présentation des 7 étapes suivies :
1. **Business Understanding** : Compréhension du contexte et des objectifs
2. **Data Understanding** : Analyse exploratoire des données
3. **Data Visualization** : Visualisation des relations entre variables
4. **Data Preparation** : Nettoyage et préparation des données
5. **Modeling** : Entraînement de plusieurs modèles
6. **Evaluation** : Comparaison des performances
7. **Deployment** : Déploiement via application web

### 2.2. Dataset utilisé
- **Source** : Dataset Heart Disease UCI
- **Taille** : 1025 patients, 14 variables
- **Variables clés** : âge, sexe, pression artérielle, cholestérol, fréquence cardiaque maximale, etc.
- **Variable cible** : target (0 ou 1)

---

## 3. Analyse Exploratoire des Données (2-3 minutes)

### 3.1. Statistiques descriptives
- **Dimensions du dataset** : 1025 lignes, 14 colonnes
- **Types de données** : 13 entiers, 1 flottant
- **Valeurs manquantes** : Aucune
- **Doublons** : 723 détectés et supprimés

### 3.2. Distribution de la variable cible
- **Classe 0 (pas de maladie)** : 499 patients (48.7%)
- **Classe 1 (maladie)** : 526 patients (51.3%)
- **Conclusion** : Classes équilibrées, pas besoin de techniques de rééquilibrage

### 3.3. Visualisations clés
- **Distribution de l'âge** : Tranche d'âge principale 48-61 ans
- **Matrice de corrélation** : Identification des variables les plus corrélées avec target
- **Boxplots comparatifs** : Âge, fréquence cardiaque maximale, cholestérol selon target
- **Countplots** : Type de douleur thoracique et sexe selon target

### 3.4. Insights principaux
- Certaines variables montrent des différences significatives entre les classes
- Le type de douleur thoracique (cp) semble particulièrement discriminant
- La fréquence cardiaque maximale (thalach) diffère entre les classes

---

## 4. Préparation des Données (1-2 minutes)

### 4.1. Nettoyage
- **Suppression des doublons** : 723 doublons supprimés
- **Dataset final** : 302 patients après nettoyage

### 4.2. Séparation features/target
- **X** : 13 variables explicatives
- **y** : Variable cible (target)

### 4.3. Division Train/Test
- **Train** : 80% des données (242 patients)
- **Test** : 20% des données (60 patients)
- **Stratification** : Conservation de la proportion des classes

### 4.4. Normalisation
- **Méthode** : StandardScaler (moyenne = 0, écart-type = 1)
- **Pourquoi** : Nécessaire pour les modèles sensibles à l'échelle (Logistic Regression, KNN, SVM)
- **Fit sur train, transform sur test** : Évite la fuite de données

---

## 5. Modélisation (2-3 minutes)

### 5.1. Modèles testés
1. **Logistic Regression**
   - Modèle linéaire simple et interprétable
   - Fonction sigmoïde pour la probabilité
   - Baseline pour la classification

2. **Random Forest**
   - Modèle d'ensemble basé sur 100 arbres de décision
   - Robuste et capable de capturer des relations non linéaires
   - Vote majoritaire pour la prédiction

3. **K-Nearest Neighbors (KNN)**
   - Basé sur la proximité (K=5 voisins)
   - Modèle simple et intuitif
   - Sensible à l'échelle des variables

4. **Support Vector Machine (SVM)**
   - Cherche l'hyperplan optimal séparant les classes
   - Noyau RBF pour gérer les relations non linéaires
   - Maximise la marge entre les classes

### 5.2. Entraînement
- Tous les modèles entraînés sur les données normalisées
- Random state fixé à 42 pour la reproductibilité
- Hyperparamètres de base utilisés

---

## 6. Évaluation des Modèles (2-3 minutes)

### 6.1. Métriques utilisées
- **Accuracy** : Proportion de prédictions correctes
- **Precision** : Proportion de vrais positifs parmi les prédictions positives
- **Recall** : Proportion de vrais positifs parmi les réels positifs
- **F1-score** : Moyenne harmonique de precision et recall

### 6.2. Pourquoi le F1-score ?
- **Contexte médical** : Important de minimiser à la fois les faux positifs et faux négatifs
- **Équilibre** : F1-score offre un meilleur équilibre que l'accuracy
- **Relevance** : Pénalise les erreurs dans les deux directions

### 6.3. Résultats comparatifs
*(Afficher le tableau des résultats)*

| Modèle | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| [À compléter après exécution] | - | - | - | - |

### 6.4. Meilleur modèle
- **Sélection** : Basée sur le F1-score le plus élevé
- **Justification** : Meilleur équilibre precision/recall pour le contexte médical
- **Matrice de confusion** : Analyse des TP, TN, FP, FN

---

## 7. Déploiement (1-2 minutes)

### 7.1. Sauvegarde des artefacts
- **model.pkl** : Meilleur modèle sauvegardé
- **scaler.pkl** : Scaler sauvegardé pour normalisation
- **Pourquoi** : Nécessaires pour l'application Flask

### 7.2. Application Flask
- **Framework** : Flask pour le déploiement web
- **Fonctionnalités** :
  - Formulaire interactif pour saisir les données du patient
  - Normalisation des données avec le scaler sauvegardé
  - Prédiction avec le modèle sauvegardé
  - Affichage du résultat et de la probabilité

### 7.3. Démonstration
- *(Si possible) : Montrer l'application en action*
- Saisie des données d'un patient exemple
- Affichage de la prédiction

---

## 8. Conclusion (1 minute)

### 8.1. Résumé des accomplissements
- ✅ Analyse complète du dataset médical
- ✅ Identification des variables importantes
- ✅ Comparaison de 4 modèles de classification
- ✅ Sélection du meilleur modèle selon le F1-score
- ✅ Déploiement via application web interactive

### 8.2. Limites
- Dataset relativement petit après nettoyage des doublons
- Hyperparamètres non optimisés (grid search non effectué)
- Validation croisée non implémentée

### 8.3. Améliorations possibles
- **Optimisation des hyperparamètres** : GridSearchCV ou RandomizedSearchCV
- **Validation croisée** : k-fold cross-validation pour une évaluation plus robuste
- **Feature engineering** : Création de nouvelles variables
- **Autres modèles** : XGBoost, LightGBM, réseaux de neurones
- **Interprétabilité** : SHAP values pour expliquer les prédictions

### 8.4. Ouverture
- Questions de l'audience

---

## Annexes

### Diapositives suggérées
1. Titre et auteur
2. Contexte et problématique
3. Objectifs du projet
4. Processus CRISP-DM
5. Dataset (statistiques)
6. Visualisations (distribution âge, matrice corrélation)
7. Préparation des données
8. Modèles testés
9. Résultats comparatifs (tableau)
10. Meilleur modèle et matrice de confusion
11. Application Flask (capture d'écran)
12. Conclusion et perspectives
13. Questions

### Points clés à retenir
- **Approche méthodologique** : Suivi rigoureux du processus CRISP-DM
- **Choix du modèle** : Basé sur le F1-score pour le contexte médical
- **Déploiement** : Application web fonctionnelle pour une utilisation pratique
- **Reproductibilité** : Code structuré et documenté

---

## Conseils pour la présentation

- **Parler lentement et clairement**
- **Utiliser des visuels** : Captures d'écran du notebook et de l'application
- **Se concentrer sur la valeur ajoutée** : Pourquoi ce projet est utile ?
- **Anticiper les questions** : Préparer des réponses pour les questions techniques
- **Montrer l'application** : Démonstration en direct si possible
- **Gérer le temps** : Respecter le temps alloué pour chaque section
