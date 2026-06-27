# Questions pour la Validation Orale - Projet Machine Learning

Ce document contient les questions potentielles qui pourraient être posées lors de la validation orale, avec leurs réponses détaillées.

---

## Table des matières

1. [Questions sur le contexte et la problématique](#contexte)
2. [Questions sur les données](#donnees)
3. [Questions sur la préparation des données](#preparation)
4. [Questions sur la modélisation](#modelisation)
5. [Questions sur l'évaluation](#evaluation)
6. [Questions sur le déploiement](#deploiement)
7. [Questions techniques avancées](#avancees)
8. [Questions sur les limites et améliorations](#limites)

---

<a name="contexte"></a>
## 1. Questions sur le contexte et la problématique

### Q1.1 : Pourquoi avoir choisi ce sujet de prédiction de maladie cardiaque ?

**Réponse :** 
J'ai choisi ce sujet car les maladies cardiovasculaires représentent un enjeu de santé publique majeur. La détection précoce du risque cardiaque permet de mettre en place des mesures préventives qui peuvent sauver des vies. De plus, ce problème de classification binaire est idéal pour illustrer l'application complète du processus de Data Science, de l'analyse exploratoire au déploiement d'un modèle en production.

### Q1.2 : Pourquoi le Machine Learning est-il pertinent dans ce contexte médical ?

**Réponse :**
Le Machine Learning permet d'analyser automatiquement les relations complexes entre plusieurs variables médicales. Contrairement aux méthodes traditionnelles basées sur des règles simples ou des scores de risque manuels, les modèles de ML peuvent capturer des patterns non évidents et fournir des prédictions personnalisées pour chaque patient. De plus, une fois le modèle entraîné, il peut effectuer des prédictions instantanées, ce qui est particulièrement utile en milieu clinique.

### Q1.3 : Quel est l'objectif principal de ce projet ?

**Réponse :**
L'objectif principal est de construire un modèle de Machine Learning capable de prédire si un patient présente un risque de maladie cardiaque en se basant sur des caractéristiques médicales simples. Le projet vise également à comparer plusieurs algorithmes de classification, sélectionner le meilleur modèle selon des métriques appropriées, et le déployer via une application web interactive pour une utilisation pratique.

---

<a name="donnees"></a>
## 2. Questions sur les données

### Q2.1 : D'où provient le dataset utilisé ?

**Réponse :**
Le dataset provient du repository UCI Machine Learning (Heart Disease Dataset). C'est un dataset public largement utilisé dans la communauté scientifique pour des recherches sur les maladies cardiovasculaires. Il contient des données réelles de patients collectées dans plusieurs hôpitaux.

### Q2.2 : Combien de patients et de variables contient le dataset ?

**Réponse :**
Le dataset original contient 1025 patients et 14 variables. Après suppression des doublons (723 détectés), le dataset final contient 302 patients uniques. Les 14 variables incluent 13 variables explicatives (âge, sexe, pression artérielle, cholestérol, etc.) et 1 variable cible (target).

### Q2.3 : Pourquoi avoir supprimé les doublons ?

**Réponse :**
Les doublons peuvent biaiser l'entraînement des modèles en donnant plus de poids à certaines observations. Si un même patient apparaît plusieurs fois, le modèle pourrait surapprendre ce profil spécifique et ne pas généraliser correctement. La suppression des doublons garantit que chaque observation est unique et représentative.

### Q2.4 : Y a-t-il des valeurs manquantes dans le dataset ?

**Réponse :**
Non, le dataset ne contient aucune valeur manquante. Toutes les 14 variables sont complètes pour chaque patient. Cela simplifie la préparation des données car aucune imputation n'est nécessaire.

### Q2.5 : Les classes sont-elles équilibrées ?

**Réponse :**
Oui, les classes sont relativement équilibrées. Dans le dataset original, il y a 526 patients avec maladie cardiaque (51.3%) et 499 sans (48.7%). Après suppression des doublons, la proportion reste similaire. Cet équilibre est important car il évite le besoin de techniques de rééquilibrage comme le suréchantillonnage ou le sous-échantillonnage.

### Q2.6 : Quelle est la signification de la variable cible ?

**Réponse :**
La variable cible `target` est binaire :
- `target = 0` : Le patient ne présente pas de maladie cardiaque
- `target = 1` : Le patient présente un risque de maladie cardiaque

C'est une variable de classification binaire classique.

---

<a name="preparation"></a>
## 3. Questions sur la préparation des données

### Q3.1 : Pourquoi avoir normalisé les données ?

**Réponse :**
La normalisation est essentielle pour les modèles sensibles à l'échelle des variables comme Logistic Regression, KNN et SVM. Sans normalisation, les variables avec de grandes valeurs (ex: cholestérol ~250) auraient plus d'influence que celles avec de petites valeurs (ex: sexe 0/1), ce qui biaiserait le modèle. La normalisation met toutes les variables sur la même échelle (moyenne = 0, écart-type = 1).

### Q3.2 : Quelle méthode de normalisation avez-vous utilisée et pourquoi ?

**Réponse :**
J'ai utilisé StandardScaler de scikit-learn, qui applique la standardisation (z-score normalization). Cette méthode soustrait la moyenne et divise par l'écart-type pour chaque variable. J'ai choisi cette méthode car elle est robuste et fonctionne bien pour la plupart des modèles de Machine Learning, particulièrement ceux basés sur les distances (KNN) ou les gradients (Logistic Regression, SVM).

### Q3.3 : Pourquoi avoir fait fit_transform sur le train et transform sur le test ?

**Réponse :**
C'est une pratique cruciale pour éviter la "fuite de données" (data leakage). Le scaler doit apprendre les paramètres (moyenne et écart-type) uniquement sur les données d'entraînement, puis appliquer ces mêmes paramètres aux données de test. Si nous faisions fit_transform sur le test, le scaler utiliserait des informations du test pour la normalisation, ce qui donnerait une performance artificiellement élevée lors de l'évaluation.

### Q3.4 : Pourquoi avoir utilisé une stratification lors de la division train/test ?

**Réponse :**
La stratification (`stratify=y`) garantit que la proportion des classes (0 et 1) est conservée dans les deux ensembles train et test. Sans stratification, il est possible que le split aléatoire crée des ensembles déséquilibrés, ce qui biaiserait l'évaluation. La stratification assure que l'évaluation est représentative de la distribution réelle des classes.

### Q3.5 : Pourquoi avoir choisi 80% pour l'entraînement et 20% pour le test ?

**Réponse :**
Le ratio 80/20 est un compromis classique en Machine Learning. Il permet d'avoir suffisamment de données pour entraîner le modèle (80%) tout en gardant un ensemble de test assez grand pour une évaluation fiable (20%). Ce ratio est souvent utilisé comme point de départ et peut être ajusté selon la taille du dataset.

---

<a name="modelisation"></a>
## 4. Questions sur la modélisation

### Q4.1 : Pourquoi avoir testé ces 4 modèles spécifiques ?

**Réponse :**
J'ai choisi ces 4 modèles car ils représentent différentes approches de classification :
- **Logistic Regression** : Modèle linéaire simple, sert de baseline
- **Random Forest** : Modèle d'ensemble, robuste et capable de capturer des relations non linéaires
- **KNN** : Modèle basé sur la distance, approche intuitive
- **SVM** : Modèle basé sur les marges, performant pour les problèmes de classification

Cette diversité permet de comparer différentes approches et de sélectionner la plus adaptée au problème.

### Q4.2 : Pourquoi Logistic Regression comme baseline ?

**Réponse :**
Logistic Regression est un excellent modèle de baseline car :
- Il est simple et rapide à entraîner
- Il est interprétable (les coefficients montrent l'importance de chaque variable)
- Il fonctionne bien comme point de comparaison pour évaluer si des modèles plus complexes apportent une valeur ajoutée significative

### Q4.3 : Pourquoi avoir fixé random_state à 42 ?

**Réponse :**
Le random_state fixe la graine aléatoire pour assurer la reproductibilité des résultats. Sans cela, chaque exécution du code donnerait des résultats légèrement différents, ce qui rendrait difficile la comparaison des modèles et la reproduction des résultats. La valeur 42 est une convention arbitraire (référence culturelle à "La Grande Question sur la Vie, l'Univers et le Reste").

### Q4.4 : Pourquoi avoir utilisé probability=True pour SVM ?

**Réponse :**
Le paramètre `probability=True` permet d'activer l'estimation des probabilités pour SVM. Par défaut, SVM ne fournit que des prédictions de classe (0 ou 1). Avec ce paramètre, nous pouvons obtenir la probabilité d'appartenance à chaque classe, ce qui est utile pour l'application Flask pour afficher le niveau de confiance de la prédiction.

### Q4.5 : Pourquoi n'avez-vous pas optimisé les hyperparamètres ?

**Réponse :**
C'est une limitation du projet actuel. L'optimisation des hyperparamètres (via GridSearchCV ou RandomizedSearchCV) permettrait d'améliorer les performances mais nécessite plus de temps de calcul. J'ai utilisé les hyperparamètres par défaut qui fonctionnent généralement bien pour une première approche. Une optimisation pourrait être une amélioration future du projet.

### Q4.6 : Pourquoi Random Forest avec 100 estimateurs ?

**Réponse :**
100 est une valeur courante pour le nombre d'arbres dans une Random Forest. Plus d'arbres améliorent généralement la performance mais augmentent le temps de calcul. 100 est un bon compromis entre performance et temps de calcul. Ce nombre pourrait être optimisé via une recherche d'hyperparamètres.

---

<a name="evaluation"></a>
## 5. Questions sur l'évaluation

### Q5.1 : Pourquoi avoir utilisé ces 4 métriques spécifiques ?

**Réponse :**
Chaque métrique apporte une information différente :
- **Accuracy** : Performance globale, mais peut être trompeuse si les classes sont déséquilibrées
- **Precision** : Important pour minimiser les faux positifs (ne pas alarmer inutilement les patients sains)
- **Recall** : Crucial pour minimiser les faux négatifs (ne pas manquer les patients malades)
- **F1-score** : Moyenne harmonique de precision et recall, offre un équilibre

Dans un contexte médical, il est important de regarder toutes ces métriques et pas seulement l'accuracy.

### Q5.2 : Pourquoi avoir choisi le F1-score pour sélectionner le meilleur modèle ?

**Réponse :**
Le F1-score est particulièrement pertinent dans un contexte médical car :
- Il pénalise à la fois les faux positifs et les faux négatifs
- Il offre un meilleur équilibre que l'accuracy lorsque les classes sont légèrement déséquilibrées
- Il permet de choisir un modèle qui performe bien sur les deux aspects critiques : détecter les patients à risque (recall) et ne pas alarmer inutilement les patients sains (precision)

### Q5.3 : Qu'est-ce que la matrice de confusion et comment l'interprétez-vous ?

**Réponse :**
La matrice de confusion est un tableau qui montre le nombre de :
- **Vrais Négatifs (TN)** : Patients sains correctement prédits comme sains
- **Faux Positifs (FP)** : Patients sains incorrectement prédits comme malades
- **Faux Négatifs (FN)** : Patients malades incorrectement prédits comme sains
- **Vrais Positifs (TP)** : Patients malades correctement prédits comme malades

Dans un contexte médical, les faux négatifs sont particulièrement problématiques car ils correspondent à des patients malades qui ne sont pas détectés par le modèle.

### Q5.4 : Pourquoi le recall est-il particulièrement important dans ce contexte ?

**Réponse :**
Le recall mesure la capacité du modèle à détecter tous les patients réellement malades. Dans un contexte médical, manquer un patient malade (faux négatif) peut avoir des conséquences graves car le patient ne recevra pas le traitement nécessaire. C'est pourquoi un recall élevé est souvent priorisé, même si cela peut légèrement augmenter les faux positifs.

### Q5.5 : Pourquoi la precision est-elle aussi importante ?

**Réponse :**
La precision mesure la proportion de patients prédits comme malades qui le sont réellement. Une precision faible signifie beaucoup de faux positifs, ce qui peut entraîner :
- Anxiété inutile pour les patients
- Examens médicaux supplémentaires coûteux
- Perte de confiance dans le système

Un équilibre entre precision et recall est donc nécessaire.

---

<a name="deploiement"></a>
## 6. Questions sur le déploiement

### Q6.1 : Pourquoi avoir choisi Flask pour le déploiement ?

**Réponse :**
Flask est un framework web léger et simple en Python, idéal pour des projets de Machine Learning de cette envergure. Il permet de créer rapidement une application web sans la complexité de frameworks plus lourds comme Django. De plus, Flask s'intègre facilement avec les bibliothèques Python de Data Science (scikit-learn, pandas, joblib).

### Q6.2 : Pourquoi sauvegarder le scaler avec le modèle ?

**Réponse :**
Le scaler est essentiel pour garantir la cohérence des données entre l'entraînement et la prédiction. Si nous utilisions un scaler différent ou aucune normalisation dans l'application, les données saisies seraient dans une échelle différente de celle utilisée pendant l'apprentissage, ce qui conduirait à des prédictions incorrectes. Le scaler sauvegardé contient les moyennes et écarts-types calculés sur les données d'entraînement.

### Q6.3 : Pourquoi avoir utilisé joblib pour sauvegarder le modèle ?

**Réponse :**
Joblib est optimisé pour sauvegarder des objets Python volumineux comme les modèles de Machine Learning. Il est plus efficace que pickle pour les tableaux numpy et les modèles scikit-learn. De plus, joblib est la méthode recommandée par scikit-learn pour la sérialisation des modèles.

### Q6.4 : Comment l'application gère-t-elle les nouvelles données ?

**Réponse :**
L'application Flask :
1. Récupère les données du formulaire web
2. Les convertit en un tableau numpy dans le bon ordre des features
3. Applique le scaler sauvegardé pour normaliser les données
4. Passe les données normalisées au modèle sauvegardé
5. Récupère la prédiction et la probabilité
6. Affiche le résultat à l'utilisateur

### Q6.5 : Pourquoi l'ordre des features est-il important dans l'application ?

**Réponse :**
L'ordre des features doit correspondre exactement à celui utilisé lors de l'entraînement. Si l'ordre est différent, le modèle interprétera chaque feature à la mauvaise place, conduisant à des prédictions complètement erronées. C'est pourquoi j'ai défini une constante `FEATURE_NAMES` dans l'application pour garantir l'ordre correct.

---

<a name="avancees"></a>
## 7. Questions techniques avancées

### Q7.1 : Qu'est-ce que le surapprentissage (overfitting) et comment l'avez-vous évité ?

**Réponse :**
Le surapprentissage se produit lorsque le modèle apprend trop bien les données d'entraînement, y compris le bruit, et ne généralise pas aux nouvelles données. Pour l'éviter :
- J'ai utilisé une division train/test pour évaluer la généralisation
- Les modèles comme Random Forest ont des mécanismes intégrés (bagging, limitation de la profondeur des arbres)
- La normalisation aide certains modèles à mieux généraliser
- Une validation croisée pourrait être ajoutée pour une évaluation plus robuste

### Q7.2 : Qu'est-ce que la fuite de données (data leakage) ?

**Réponse :**
La fuite de données se produit lorsque des informations du test set sont utilisées, directement ou indirectement, pour entraîner le modèle. Cela conduit à des performances artificiellement élevées lors de l'évaluation mais à de mauvaises performances en production. Pour l'éviter :
- J'ai fait fit_transform uniquement sur le train
- J'ai fait transform sur le test
- J'ai utilisé la stratification pour éviter un biais dans le split

### Q7.3 : Pourquoi n'avez-vous pas utilisé de validation croisée ?

**Réponse :**
La validation croisée (k-fold) est une technique plus robuste que le simple split train/test car elle utilise plusieurs partitions des données. Je ne l'ai pas implémentée par souci de simplicité et de temps, mais ce serait une amélioration importante pour obtenir une évaluation plus fiable des performances des modèles.

### Q7.4 : Qu'est-ce que le kernel RBF utilisé dans SVM ?

**Réponse :**
RBF (Radial Basis Function) est un kernel non linéaire qui permet à SVM de capturer des relations complexes entre les variables. Contrairement au kernel linéaire qui ne peut séparer les données que par un hyperplan droit, le kernel RBF projette les données dans un espace de dimension supérieure où elles deviennent linéairement séparables. C'est particulièrement utile quand les classes ne sont pas séparables linéairement dans l'espace original.

### Q7.5 : Comment fonctionne KNN ?

**Réponse :**
KNN (K-Nearest Neighbors) est un algorithme basé sur l'apprentissage instance. Pour prédire la classe d'une nouvelle observation :
1. Il calcule la distance entre cette observation et toutes les observations du training set
2. Il sélectionne les K observations les plus proches
3. Il attribue la classe majoritaire parmi ces K voisins

C'est un algorithme simple mais qui peut être performant si les features sont bien normalisées et si K est bien choisi.

---

<a name="limites"></a>
## 8. Questions sur les limites et améliorations

### Q8.1 : Quelles sont les principales limites de votre projet ?

**Réponse :**
Les principales limites sont :
- **Taille du dataset** : Après suppression des doublons, seulement 302 patients, ce qui est relativement petit
- **Hyperparamètres** : Non optimisés, utilisation des valeurs par défaut
- **Validation** : Pas de validation croisée, seulement un simple split train/test
- **Feature engineering** : Aucune création de nouvelles features
- **Interprétabilité** : Pas d'analyse de l'importance des features (feature importance, SHAP values)

### Q8.2 : Quelles améliorations pourraient être apportées ?

**Réponse :**
Plusieurs améliorations sont possibles :
- **Optimisation des hyperparamètres** : GridSearchCV ou RandomizedSearchCV
- **Validation croisée** : k-fold cross-validation pour une évaluation plus robuste
- **Feature engineering** : Création de nouvelles features (ex: BMI, catégories d'âge)
- **Autres modèles** : XGBoost, LightGBM, réseaux de neurones
- **Interprétabilité** : SHAP values ou LIME pour expliquer les prédictions
- **Plus de données** : Augmenter la taille du dataset si possible
- **Déploiement cloud** : Déployer l'application sur un service cloud (AWS, Heroku, etc.)

### Q8.3 : Pourquoi n'avez-vous pas fait de feature engineering ?

**Réponse :**
Le feature engineering n'a pas été priorisé dans ce projet pour garder une approche simple et directe. Les variables existantes du dataset sont déjà pertinentes pour le problème médical. Cependant, le feature engineering pourrait être une amélioration intéressante, par exemple en créant des catégories d'âge ou en calculant l'IMC si la taille et le poids étaient disponibles.

### Q8.4 : Pourquoi n'avez-vous pas analysé l'importance des features ?

**Réponse :**
L'analyse de l'importance des features (feature importance via Random Forest ou coefficients de Logistic Regression) n'a pas été incluse mais serait très utile pour comprendre quelles variables sont les plus déterminantes pour la prédiction. Cela pourrait être ajouté facilement et apporterait une valeur interprétative importante au projet.

### Q8.5 : Comment évalueriez-vous la performance du modèle en conditions réelles ?

**Réponse :**
Pour évaluer la performance en conditions réelles :
- **Test sur de nouvelles données** : Collecter de nouveaux patients et comparer les prédictions aux diagnostics réels
- **Monitoring continu** : Suivre les performances au fil du temps pour détecter une dégradation (concept drift)
- **Feedback des médecins** : Obtenir l'avis d'experts médicaux sur la pertinence des prédictions
- **A/B testing** : Comparer les décisions médicales avec et sans l'aide du modèle

---

## Conseils pour répondre aux questions

1. **Soyez honnête** : Si vous ne savez pas, dites-le et proposez de chercher
2. **Soyez concis** : Allez à l'essentiel sans trop détailler
3. **Reliez au contexte** : Expliquez pourquoi c'est important pour ce projet spécifique
4. **Montrez votre compréhension** : Utilisez les termes techniques correctement
5. **Soyez structuré** : Organisez votre réponse en points clairs
6. **Anticipez** : Préparez des exemples concrets pour illustrer vos réponses

---

## Questions pièges possibles

### Q : "Pourquoi votre modèle n'a-t-il pas 100% d'accuracy ?"

**Réponse :** Un modèle avec 100% d'accuracy est généralement le signe d'un surapprentissage ou d'une erreur dans l'évaluation. Dans le monde réel, les données médicales sont complexes et il est normal qu'un modèle ne soit pas parfait. L'important est d'avoir un modèle qui généralise bien et qui offre un bon équilibre entre precision et recall.

### Q : "Pourquoi ne pas avoir utilisé un réseau de neurones ?"

**Réponse :** Les réseaux de neurones sont puissants mais nécessitent généralement plus de données et plus de temps d'entraînement. Pour ce dataset de taille modérée, les modèles classiques comme Random Forest ou SVM sont souvent plus appropriés et offrent de bonnes performances avec moins de complexité.

### Q : "Votre modèle est-il prêt à être utilisé en milieu clinique ?"

**Réponse :** Non, ce projet est une démonstration technique. Pour un usage clinique réel, il faudrait :
- Validation sur des données externes (multi-centres)
- Approbation réglementaire
- Tests cliniques
- Intégration avec les systèmes médicaux existants
- Formation du personnel médical

Ce projet montre le potentiel mais n'est pas un produit final pour un usage médical.
