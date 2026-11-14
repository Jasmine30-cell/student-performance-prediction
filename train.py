import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# PHASE 1: LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/student-por.csv", sep=";")

print("\nFIRST FIVE STUDENT RECORDS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())


# ==========================================
# PHASE 2: DATA CLEANING
# ==========================================

df = df.drop_duplicates()

print("\nDATASET AFTER REMOVING DUPLICATES")
print(df.shape)

print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum().sum())

# Fill numerical missing values with median
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum().sum())


# ==========================================
# PHASE 3: FEATURE IDENTIFICATION
# ==========================================

numerical_columns = df.select_dtypes(
    include=np.number
).columns

categorical_columns = df.select_dtypes(
    include="object"
).columns

print("\nNUMERICAL COLUMNS")
print(list(numerical_columns))

print("\nCATEGORICAL COLUMNS")
print(list(categorical_columns))


# ==========================================
# CREATE PERFORMANCE CATEGORIES
# ==========================================

def categorize_performance(grade):

    if grade <= 9:
        return "Low"

    elif grade <= 14:
        return "Medium"

    else:
        return "High"


df["Performance"] = df["G3"].apply(categorize_performance)

print("\nPERFORMANCE CATEGORY DISTRIBUTION")
print(df["Performance"].value_counts())

print("\nFIRST FIVE STUDENTS WITH PERFORMANCE CATEGORY")
print(df[["G3", "Performance"]].head())


# ==========================================
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# Graph 1: Distribution of Final Grades

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="G3", bins=20, kde=True)

plt.title("Distribution of Student Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.show()


# Graph 2: Study Time vs Performance

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="studytime", y="G3")

plt.title("Study Time vs Final Academic Performance")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")

plt.show()


# Graph 3: Absences vs Final Grade

plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x="absences", y="G3")

plt.title("School Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.show()


# Graph 4: Previous Failures vs Final Grade

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="failures", y="G3")

plt.title("Previous Failures vs Final Grade")
plt.xlabel("Number of Previous Failures")
plt.ylabel("Final Grade (G3)")

plt.show()


# Graph 5: Correlation Heatmap

plt.figure(figsize=(14, 10))

correlation = df.select_dtypes(include=np.number).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Student Attributes")

plt.show()


# ==========================================
# PHASE 5: FEATURE SELECTION
# ==========================================

# Input features
X = df.drop(["G3", "Performance"], axis=1)

# Target variable
y = df["Performance"]

print("\nINPUT FEATURES")
print(X.head())

print("\nTARGET VARIABLE")
print(y.head())

print("\nNUMBER OF INPUT FEATURES")
print(X.shape[1])

print("\nTARGET DISTRIBUTION")
print(y.value_counts())


# Identify numerical and categorical features

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()

print("\nNUMERICAL FEATURES")
print(numerical_features)

print("\nCATEGORICAL FEATURES")
print(categorical_features)


# ==========================================
# SPLIT DATASET
# ==========================================

from sklearn.model_selection import train_test_split

# 70% training, 30% remaining

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 15% validation, 15% testing

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDATASET SPLIT RESULTS")

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)


# ==========================================
# DATA PREPROCESSING
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),

        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# Fit on training data only

X_train_processed = preprocessor.fit_transform(X_train)

# Transform validation and testing data

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)

print("\nPREPROCESSING COMPLETED")

print("Training shape:", X_train_processed.shape)
print("Validation shape:", X_val_processed.shape)
print("Testing shape:", X_test_processed.shape)


# ==========================================
# TARGET ENCODING
# ==========================================

target_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

y_train_encoded = y_train.map(target_mapping)
y_val_encoded = y_val.map(target_mapping)
y_test_encoded = y_test.map(target_mapping)

print("\nENCODED TARGET VALUES")
print(y_train_encoded.head())


# ==========================================
# CONVERT DATA INTO NUMPY ARRAYS
# ==========================================

if hasattr(X_train_processed, "toarray"):

    X_train_processed = X_train_processed.toarray()
    X_val_processed = X_val_processed.toarray()
    X_test_processed = X_test_processed.toarray()


X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)

X_val_processed = np.asarray(
    X_val_processed, dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

print("\nFINAL DATA SHAPES")

print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Testing:", X_test_processed.shape)


# ==========================================
# PHASE 6: BUILD DEEP NEURAL NETWORK
# ==========================================

model = tf.keras.Sequential([

    tf.keras.Input(
        shape=(X_train_processed.shape[1],)
    ),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(3, activation="softmax")

])

model.summary()


# ==========================================
# PHASE 7: COMPILE THE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL COMPILED SUCCESSFULLY!")


# ==========================================
# PHASE 8: TRAIN WITH EARLY STOPPING
# ==========================================

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True

)

history = model.fit(

    X_train_processed,

    y_train_encoded,

    validation_data=(
        X_val_processed,
        y_val_encoded
    ),

    epochs=100,

    batch_size=32,

    callbacks=[early_stop]

)

print("\nMODEL TRAINING COMPLETED WITH EARLY STOPPING!")


# ==========================================
# PHASE 9: MODEL EVALUATION
# ==========================================

y_pred_prob = model.predict(X_test_processed)

y_pred = np.argmax(y_pred_prob, axis=1)

test_accuracy = accuracy_score(
    y_test_encoded,
    y_pred
)

print("\nTEST ACCURACY:", test_accuracy)

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test_encoded,
    y_pred,
    target_names=["Low", "Medium", "High"],
    zero_division=0
))

print("\nCONFUSION MATRIX")

print(confusion_matrix(
    y_test_encoded,
    y_pred
))


# ==========================================
# PHASE 10: TRAINING AND VALIDATION GRAPHS
# ==========================================

# Accuracy Graph

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Model Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid()

plt.show()


# Loss Graph

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Model Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid()

plt.show()


# ==========================================
# PHASE 12: RANDOM FOREST MODEL
# ==========================================

print("\nRANDOM FOREST MODEL TRAINING")

# Create Random Forest model

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train the model

rf_model.fit(
    X_train_processed,
    y_train_encoded
)

print("\nRANDOM FOREST TRAINING COMPLETED!")


# Make predictions

rf_predictions = rf_model.predict(X_test_processed)


# Evaluate Random Forest

rf_accuracy = accuracy_score(
    y_test_encoded,
    rf_predictions
)

print("\nRANDOM FOREST TEST ACCURACY:")
print(rf_accuracy)

print("\nRANDOM FOREST CLASSIFICATION REPORT")

print(classification_report(
    y_test_encoded,
    rf_predictions,
    target_names=["Low", "Medium", "High"],
    zero_division=0
))


# ==========================================
# MODEL COMPARISON
# ==========================================

print("\nDNN VS RANDOM FOREST COMPARISON")

print("DNN Accuracy:", test_accuracy)

print("Random Forest Accuracy:", rf_accuracy)

# ==========================================
# PHASE 13: HYPERPARAMETER TUNING
# ==========================================

print("\nHYPERPARAMETER TUNING")

# Parameters to test
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

# Create Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train using training data
grid_search.fit(
    X_train_processed,
    y_train_encoded
)

# Display best parameters
print("\nBEST PARAMETERS:")
print(grid_search.best_params_)

# Display cross-validation accuracy
print("\nBEST CROSS-VALIDATION ACCURACY:")
print(grid_search.best_score_)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Predict using test data
tuned_predictions = best_rf_model.predict(
    X_test_processed
)

# Evaluate tuned model
tuned_accuracy = accuracy_score(
    y_test_encoded,
    tuned_predictions
)

print("\nTUNED RANDOM FOREST TEST ACCURACY:")
print(tuned_accuracy)

print("\nTUNED RANDOM FOREST CLASSIFICATION REPORT:")
print(classification_report(
    y_test_encoded,
    tuned_predictions,
    target_names=["Low", "Medium", "High"],
    zero_division=0
))

# ==========================================
# PHASE 14: SAVE TRAINED MODELS
# ==========================================

print("\nSAVING TRAINED MODELS")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save DNN model
model.save("models/student_performance_model.keras")

# Save tuned Random Forest model
joblib.dump(
    best_rf_model,
    "models/random_forest_model.pkl"
)

# Save preprocessing object
joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

# Save target mapping
joblib.dump(
    target_mapping,
    "models/target_mapping.pkl"
)

print("\nALL MODELS AND PREPROCESSORS SAVED SUCCESSFULLY!")