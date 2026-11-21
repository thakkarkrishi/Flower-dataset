
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv("D:/data science/flower dataset/Iris.csv")

# 2. Remove Id Column
df.drop('Id', axis=1, inplace=True)

# 3. EDA
print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Description:")
print(df.describe())

print("\nSpecies Count:")
print(df['Species'].value_counts())

# 4. Handle Missing Values
df.dropna(inplace=True)

# 5. Label Encoding
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

print("\nEncoded Species:")
print(le.classes_)

# 6. Separate X and Y
X = df.drop('Species', axis=1)
y = df['Species']

# 7. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 8. Classification Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "KNN": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    "SVM": make_pipeline(
        StandardScaler(), SVC()
    ),
    "Naive Bayes": GaussianNB()
}

# 9. Train and Evaluate Models
results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print("\n", name)
    print("Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_
    ))

# 10. Model Accuracy Comparison Graph
plt.figure(figsize=(9, 5))

plt.bar(results.keys(), results.values())

plt.title("Model Accuracy Comparison")
plt.xlabel("Classification Models")
plt.ylabel("Accuracy")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 11. Flower Features Comparison Graph
features = [
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]

feature_data = df.copy()
feature_data['Species'] = le.inverse_transform(feature_data['Species'])

feature_data.groupby('Species')[features].mean().plot(
    kind='bar', figsize=(10, 6)
)

plt.title("Average Flower Features Comparison")
plt.xlabel("Species")
plt.ylabel("Average Measurement (cm)")

plt.legend(title="Flower Features")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 12. Select Best Model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", results[best_model_name])

# 13. Save Best Model using Pickle
with open("D:/data science/flower dataset/iris_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# 14. Save Label Encoder
with open("D:/data science/flower dataset/iris_encoder.pkl", "wb") as file:
    pickle.dump(le, file)

print("\nBest Model and Label Encoder Saved Successfully!")
