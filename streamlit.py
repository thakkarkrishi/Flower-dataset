
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Page Configuration

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="wide"
)

# 2. Custom CSS

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
    color: black;
}

h1, h2, h3 {
    color: #172554 !important;
}

p, label, span {
    color: #222222;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
}

.stButton button {
    background-color: #3949ab;
    color: white;
    height: 45px;
    width: 100%;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# 3. Load Dataset

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

df = X.copy()
df["Species"] = [iris.target_names[i] for i in y]

features = iris.feature_names

# 4. Train KNN Model

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# 5. Header

st.markdown(
    "<h1 style='text-align:center;'>🌸 Iris Flower Classification</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Machine Learning Application using KNN Algorithm</p>",
    unsafe_allow_html=True
)

st.divider()

# 6. Dataset Information

st.header("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Total Features", 4)
col3.metric("Flower Species", 3)

with st.expander("View Iris Dataset"):
    st.dataframe(df, width="stretch")

# 7. Exploratory Data Analysis

st.header("Exploratory Data Analysis")

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(),
    width="stretch"
)

st.subheader("Missing Values")

st.dataframe(
    df.isnull().sum().rename("Missing Values"),
    width="stretch"
)

# Graph 1: Species Distribution

st.subheader("Species Distribution")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    sns.countplot(
        data=df,
        x="Species",
        hue="Species",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Flower Count")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Graph 2: Feature Distribution

with col2:

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for feature in features:

        sns.kdeplot(
            data=df,
            x=feature,
            ax=ax,
            label=feature
        )

    ax.set_title("Feature Distribution")
    ax.set_xlabel("Measurement (cm)")
    ax.legend(fontsize=6)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Graph 3: Outlier Detection

st.subheader("Outlier Detection")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    sns.boxplot(
        data=df,
        x="Species",
        y="sepal length (cm)",
        hue="Species",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Sepal Length")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    sns.boxplot(
        data=df,
        x="Species",
        y="petal length (cm)",
        hue="Species",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Petal Length")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# 8. Model Performance

st.header("Model Performance")

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

col1, col2 = st.columns(2)

# Correlation Heatmap

with col1:

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        X.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 9},
        ax=ax
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Confusion Matrix

with col2:

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        annot_kws={"size": 9},
        ax=ax
    )

    ax.set_xlabel("Predicted Species")
    ax.set_ylabel("Actual Species")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.write("**Algorithm:** K-Nearest Neighbors")
st.write("**K Value:** 5")
st.write("**Test Size:** 20%")
st.write("**Training Records:**", len(X_train))
st.write("**Testing Records:**", len(X_test))

# 9. Classification Report

with st.expander("View Classification Report"):

    report = classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(2),
        width="stretch"
    )

st.divider()

# 10. Flower Prediction

st.header("Predict Iris Flower Species")

st.write("Enter flower measurements in centimeters.")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

with col2:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

# 11. Prediction

if st.button("Predict Flower Species"):

    new_flower = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=iris.feature_names
    )

    prediction = model.predict(new_flower)

    species = iris.target_names[prediction[0]]

    st.markdown(
        f"""
        <div style="
            background-color: #d1fae5;
            padding: 12px;
            border-radius: 10px;
            text-align: center;
            margin-top: 15px;
        ">
            <h3 style="color: #065f46 !important; margin: 5px;">
                Predicted Flower
            </h3>
            <h2 style="color: #047857 !important; margin: 5px;">
                {species.upper()}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# 12. Footer

st.divider()

st.markdown(
    "<p style='text-align:center;'>Iris Flower Classification | KNN Machine Learning Project</p>",
    unsafe_allow_html=True
)
