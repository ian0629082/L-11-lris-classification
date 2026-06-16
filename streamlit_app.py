"""
Iris Classifier — CRISP-DM Streamlit App
=========================================
Interactive 3D visualizations + live predictions.
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Page config ──
st.set_page_config(page_title="Iris Classifier (CRISP-DM)", layout="wide")
st.title("🌸 Iris Species Classifier")
st.markdown("Following **CRISP-DM** methodology — _Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Deployment_")

# ── Load & cache data ──
@st.cache_data
def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

X, y, feature_names, target_names = load_data()

# ── Train & cache model ──
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Candidate models
    models = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=200),
        "RandomForest": RandomForestClassifier(random_state=42),
    }
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
        cv_results[name] = (scores.mean(), scores.std())

    # Tune SVM
    grid = GridSearchCV(
        SVC(random_state=42),
        {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1, "scale", "auto"], "kernel": ["rbf"]},
        cv=5, scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_train_s, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    return scaler, best_model, grid.best_params_, grid.best_score_, test_acc, cm, report, cv_results, X_train, X_test, y_train, y_test

scaler, best_model, best_params, best_cv_acc, test_acc, cm, report, cv_results, X_train, X_test, y_train, y_test = train_model()

# ──────────────────────────────
# PHASE 1 & 2: Business / Data Understanding
# ──────────────────────────────
with st.expander("📋 Phase 1 & 2 — Business & Data Understanding", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", X.shape[0])
    col2.metric("Features", X.shape[1])
    col3.metric("Classes", len(target_names))

    dist_df = {name: int((y == i).sum()) for i, name in enumerate(target_names)}
    st.write("**Class distribution:**", dist_df)
    st.write("**Feature names:**", feature_names)

    # 3D scatter
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["r", "g", "b"]
    for i, name in enumerate(target_names):
        mask = y == i
        ax.scatter(X[mask, 2], X[mask, 3], X[mask, 0],
                   c=colors[i], label=name, s=40, edgecolors="k", alpha=0.8)
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_zlabel("Sepal length (cm)")
    ax.set_title("Iris Dataset — 3D Scatter")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ──────────────────────────────
# PHASE 3: Data Preparation
# ──────────────────────────────
with st.expander("🔧 Phase 3 — Data Preparation", expanded=False):
    st.write(f"**Split:** 80% train ({X_train.shape[0]}), 20% test ({X_test.shape[0]})")
    st.write("**Scaler:** StandardScaler (zero mean, unit variance)")

# ──────────────────────────────
# PHASE 4: Modelling
# ──────────────────────────────
with st.expander("🤖 Phase 4 — Modelling", expanded=True):
    st.subheader("Candidate Models — 5-Fold CV Accuracy")
    cv_df = {name: f"{mean:.4f} (±{std:.4f})" for name, (mean, std) in cv_results.items()}
    st.table([{"Model": k, "CV Accuracy": v} for k, v in cv_df.items()])

    st.subheader("Best Model: SVM (Tuned)")
    st.write(f"**Best params:** `{best_params}`")
    st.write(f"**Best CV accuracy:** {best_cv_acc:.4f}")

# ──────────────────────────────
# PHASE 5: Evaluation
# ──────────────────────────────
with st.expander("📊 Phase 5 — Evaluation", expanded=True):
    col1, col2 = st.columns(2)
    col1.metric("Test Accuracy", f"{test_acc:.4f}")
    col2.metric("Success (>=95%)", "✅ PASSED" if test_acc >= 0.95 else "❌ NOT PASSED")

    st.subheader("Classification Report")
    report_df = []
    for cls in target_names:
        r = report[cls]
        report_df.append({"Class": cls, "Precision": f"{r['precision']:.3f}",
                          "Recall": f"{r['recall']:.3f}", "F1": f"{r['f1-score']:.3f}",
                          "Support": int(r['support'])})
    st.table(report_df)

    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(ax=ax_cm, cmap="Blues")
    ax_cm.set_title("Confusion Matrix — Iris Test Set")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.subheader("3D Decision Surface + 2D Contour")
    x_min, x_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    y_min, y_max = X[:, 3].min() - 0.5, X[:, 3].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    sepal_w_mean = X[:, 1].mean()
    sepal_l_mean = X[:, 0].mean()
    grid_4d = np.c_[np.full(xx.ravel().shape, sepal_l_mean),
                    np.full(xx.ravel().shape, sepal_w_mean),
                    xx.ravel(), yy.ravel()]
    Z = best_model.predict(scaler.transform(grid_4d)).reshape(xx.shape)

    fig2 = plt.figure(figsize=(12, 5))
    ax1 = fig2.add_subplot(121, projection="3d")
    zz = np.full_like(xx, sepal_l_mean)
    Z_colors = np.zeros((*Z.shape, 3))
    for i, c in enumerate(colors):
        Z_colors[Z == i] = plt.matplotlib.colors.to_rgb(c)
    ax1.plot_surface(xx, yy, zz, facecolors=Z_colors, alpha=0.4, rstride=1, cstride=1)
    for i, name in enumerate(target_names):
        mask = y == i
        ax1.scatter(X[mask, 2], X[mask, 3], X[mask, 0],
                    c=colors[i], label=name, s=30, edgecolors="k", alpha=0.8)
    ax1.set_xlabel("Petal length (cm)")
    ax1.set_ylabel("Petal width (cm)")
    ax1.set_zlabel("Sepal length (cm)")
    ax1.set_title("3D Decision Surface")
    ax1.legend()

    ax2 = fig2.add_subplot(122)
    ax2.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    for i, name in enumerate(target_names):
        mask = y == i
        ax2.scatter(X[mask, 2], X[mask, 3], c=colors[i], label=name, s=30, edgecolors="k")
    ax2.set_xlabel("Petal length (cm)")
    ax2.set_ylabel("Petal width (cm)")
    ax2.set_title("2D Decision Contour")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ──────────────────────────────
# PHASE 6: Deployment — Interactive Prediction
# ──────────────────────────────
with st.expander("🚀 Phase 6 — Deployment (Live Prediction)", expanded=True):
    st.subheader("Predict Iris Species")

    col1, col2 = st.columns(2)
    with col1:
        sl = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sw = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    with col2:
        pl = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        pw = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    if st.button("🔍 Predict", type="primary"):
        sample = np.array([[sl, sw, pl, pw]])
        sample_s = scaler.transform(sample)
        pred = best_model.predict(sample_s)[0]
        proba = best_model.decision_function(sample_s)[0] if hasattr(best_model, "decision_function") else None

        species = target_names[pred]
        st.success(f"**Prediction: {species}**")

        if proba is not None:
            st.write("**Decision function values (confidence):**")
            for i, name in enumerate(target_names):
                st.write(f"  {name}: {proba[i]:.4f}")
