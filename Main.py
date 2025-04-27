##FeatureEngineering

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from sklearn.preprocessing import OneHotEncoder
# Link to get the datasets : https://drive.google.com/file/d/1t-6UwC4qYQ80cmwNRj9OtugrzNMpDaoZ/view?usp=drive_link
# Load the training and test sets
train_df = gpd.read_file('./train.geojson')
test_df = gpd.read_file('./test.geojson')

def extract_geometric_features(df):
    features = []
    for geom in df['geometry']:
        if geom.is_empty:
            features.append([0, 0, 0, 0, 0])
        else:
            area = geom.area
            perimeter = geom.length
            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            centroid_x, centroid_y = geom.centroid.x, geom.centroid.y
            features.append([area, perimeter, compactness, centroid_x, centroid_y])
    return pd.DataFrame(features, columns=["area", "perimeter", "compactness", "centroid_x", "centroid_y"])

def extract_image_features(df):
    feature_columns = []
    for i in range(1, 6):
        for color in ["red", "green", "blue"]:
            feature_columns.extend([f"img_{color}_mean_date{i}", f"img_{color}_std_date{i}"])
    return df[feature_columns] if set(feature_columns).issubset(df.columns) else pd.DataFrame()

def process_temporal_features(df):
    temporal_features = pd.DataFrame()
    status_columns = [f"change_status_date{i}" for i in range(5)]
    if status_columns:
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoded = one_hot_encoder.fit_transform(df[status_columns]).toarray()
        status_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(status_columns))
        temporal_features = pd.concat([temporal_features, status_df], axis=1)
    date_columns = [f"date{i}" for i in range(5)]
    if all(col in df.columns for col in date_columns):
        for i in range(len(date_columns) - 1):
            col_name = f"time_diff_{date_columns[i]}_{date_columns[i + 1]}"
            temporal_features[col_name] = (pd.to_datetime(df[date_columns[i + 1]], dayfirst=True) - pd.to_datetime(df[date_columns[i]], dayfirst=True)).dt.days
    return temporal_features

def process_neighbourhood_features(df):
    neighbourhood_features = pd.DataFrame()
    if 'urban_type' in df.columns:
        urban_types_df = pd.get_dummies(df['urban_type'], prefix='urban_type')
        neighbourhood_features = pd.concat([neighbourhood_features, urban_types_df], axis=1)
    if 'geography_type' in df.columns:
        geography_types_df = df['geography_type'].str.get_dummies(sep=',')
        neighbourhood_features = pd.concat([neighbourhood_features, geography_types_df], axis=1)
    return neighbourhood_features

# Extract features from the training and test sets
train_geometric_features = extract_geometric_features(train_df)
test_geometric_features = extract_geometric_features(test_df)

train_image_features = extract_image_features(train_df)
test_image_features = extract_image_features(test_df)

train_temporal_features = process_temporal_features(train_df)
test_temporal_features = process_temporal_features(test_df)

train_neighbourhood_features = process_neighbourhood_features(train_df)
test_neighbourhood_features = process_neighbourhood_features(test_df)

# Combine all features into a single DataFrame for each set
train_features = pd.concat([train_geometric_features, train_image_features, train_temporal_features, train_neighbourhood_features], axis=1)
test_features = pd.concat([test_geometric_features, test_image_features, test_temporal_features, test_neighbourhood_features], axis=1)

# Save the processed features to CSV without the target label
train_features.to_csv("train_features.csv", index=False)
test_features.to_csv("test_features.csv", index=False)

print("Feature extraction complete!")

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
##Dimensionnality Reduction (PCA)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the updated CSV without the target column
X = pd.read_csv("train_features.csv")

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Handle missing values (if any) by imputing with the mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Split the data (no target now)
X_train, X_val = train_test_split(X_imputed, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Apply PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

# Reapply PCA with the optimal number of components
n_components=0
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
#Ajouter un enregistrement des vecteurs réduits

print("PCA completed successfully!")
##----------------------------------------------------------------------------------------------------------------------------------------------------------------
##Logistic Regression
# Logistic Regression Model Training and Evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Prepare target labels (assumes 'change_type' exists in train_df)
target = train_df["change_type"]

# Encode target labels into numeric values
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Fill missing values in features if any
train_features = train_features.fillna(0)
test_features = test_features.fillna(0)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, target_encoded, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_features_scaled = scaler.transform(test_features)

# Initialize and train the Logistic Regression classifier with increased max_iter
logreg = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
logreg.fit(X_train_scaled, y_train)

# Evaluate on the validation set using both Accuracy and Mean F1-Score
val_predictions = logreg.predict(X_val_scaled)
accuracy = accuracy_score(y_val, val_predictions)
f1 = f1_score(y_val, val_predictions, average='macro')
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
print("Validation Mean F1-Score: {:.4f}".format(f1))

# ------------------------------
# Prediction on Test Data and Submission File Creation
# ------------------------------

# Predict on the test set
test_predictions = logreg.predict(test_features_scaled)

# Convert numeric predictions back to original labels
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Create the submission file (sample_submission.csv)
# The file must contain a header and follow the format:
# Id, change_type
# 1,1
submission_df = pd.DataFrame({
    "Id": test_df.index, 
    "change_type": test_predictions_labels
})
submission_df.to_csv("sample_LR_submission.csv", index=False)

print("Submission file 'sample_LR_submission.csv' created!")
##----------------------------------------------------------------------------------------------------------------------------------------------------------------
##Random Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Préparation de la variable cible (en supposant que "change_type" existe dans train_df)
target = train_df["change_type"]

# Encodage des labels cibles en valeurs numériques
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Gestion des valeurs manquantes dans les caractéristiques
train_features = train_features.fillna(0)
test_features = test_features.fillna(0)

# Séparation des données d'entraînement en sets d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(train_features, target_encoded, test_size=0.2, random_state=42)

# Initialisation et entraînement du classificateur Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Évaluation sur le jeu de validation
val_predictions = rf.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
f1 = f1_score(y_val, val_predictions, average='macro')
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
print("Validation Mean F1-Score: {:.4f}".format(f1))

# Prédictions sur le jeu de test
test_predictions = rf.predict(test_features)

# Conversion des prédictions numériques vers les étiquettes originales
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Création du fichier de soumission (sample_submission.csv)
# Le format doit être : Id, change_type
submission_df = pd.DataFrame({
    "Id": test_df.index,
    "change_type": test_predictions_labels
})
submission_df.to_csv("sample_RF_submission.csv", index=False)

print("Fichier de soumission 'sample_RF_submission.csv' créé !")
##----------------------------------------------------------------------------------------------------------------------------------------------------------------
##XGBoost
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from collections import Counter

# Encodage des labels cibles en valeurs numériques
label_encoder = LabelEncoder()
# Trouver la bonne colonne automatiquement
target_column = [col for col in train_df.columns if 'change' in col.lower()][0]
target_encoded = label_encoder.fit_transform(train_df[target_column])

# Vérification de la distribution des classes
class_distribution = Counter(target_encoded)
print(f"Distribution des classes : {class_distribution}")

# Gestion des valeurs manquantes avec imputation
imputer = SimpleImputer(strategy='median')  # Utilisation de la médiane pour l'imputation
train_features_imputed = imputer.fit_transform(train_features)
test_features_imputed = imputer.transform(test_features)

# Séparation des données en train/validation
X_train, X_val, y_train, y_val = train_test_split(train_features_imputed, target_encoded, test_size=0.2, random_state=42)

# Initialisation et entraînement du modèle XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=6, 
    random_state=42,
    gamma=0.1,  # Ajouter un peu de régularisation
    subsample=0.8,  # Utiliser un sous-ensemble des données pour chaque arbre
    colsample_bytree=0.8  # Utiliser un sous-ensemble des colonnes
)

# Validation croisée pour évaluer la performance globale du modèle
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')  # Cross-validation avec 5 plis
print(f"Validation croisée - Scores d'accuracy : {cv_scores}")
print(f"Validation croisée - Accuracy moyenne : {cv_scores.mean():.2f}")

# Entraînement du modèle sur l'ensemble d'entraînement
xgb_model.fit(X_train, y_train)

# Évaluation sur le jeu de validation
val_predictions = xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
f1 = f1_score(y_val, val_predictions, average='macro')

print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
print("Validation Mean F1-Score: {:.4f}".format(f1))

# Prédiction sur le jeu de test
test_predictions = xgb_model.predict(test_features_imputed)

# Conversion des prédictions numériques en labels originaux
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Création du fichier de soumission
submission_df = pd.DataFrame({
    "Id": test_df.index,
    "change_type": test_predictions_labels
})
submission_df.to_csv("sample_XGBoost_submission.csv", index=False)

print("Fichier de soumission 'sample_XGBoost_submission.csv' créé !")