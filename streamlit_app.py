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
import plotly.express as px
import plotly.graph_objects as go
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

# ── Colour palette ──
colors = ["#FF4B4B", "#2ECC71", "#3498DB"]

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

    # Interactive 3D scatter — Plotly
    df_iris = px.data.iris()
    fig3d = px.scatter_3d(
        df_iris,
        x="petal_length", y="petal_width", z="sepal_length",
        color="species",
        color_discrete_map={"setosa": colors[0], "versicolor": colors[1], "virginica": colors[2]},
        title="Iris Dataset — 3D Scatter",
        labels={"petal_length": "Petal length (cm)", "petal_width": "Petal width (cm)", "sepal_length": "Sepal length (cm)"},
    )
    fig3d.update_traces(marker=dict(size=5, line=dict(width=1, color="black")))
    fig3d.update_layout(height=500, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig3d, use_container_width=True)

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

    st.subheader("Interactive 3D Decision Surface")

    # Build meshgrid & predict
    x_min, x_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    y_min, y_max = X[:, 3].min() - 0.5, X[:, 3].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                         np.linspace(y_min, y_max, 30))
    sepal_w_mean = X[:, 1].mean()
    sepal_l_mean = X[:, 0].mean()
    grid_4d = np.c_[np.full(xx.ravel().shape, sepal_l_mean),
                    np.full(xx.ravel().shape, sepal_w_mean),
                    xx.ravel(), yy.ravel()]
    Z = best_model.predict(scaler.transform(grid_4d)).reshape(xx.shape)

    # Surface for each class region
    fig_surf = go.Figure()
    class_cmap = {0: colors[0], 1: colors[1], 2: colors[2]}
    for cls_id in range(3):
        mask_class = Z == cls_id
        zz_surf = np.full_like(xx, sepal_l_mean, dtype=float)
        zz_surf[~mask_class] = np.nan

        fig_surf.add_trace(go.Surface(
            x=xx, y=yy, z=zz_surf,
            opacity=0.35,
            colorscale=[[0, class_cmap[cls_id]], [1, class_cmap[cls_id]]],
            showscale=False,
            name=target_names[cls_id],
            contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        ))

    # Overlay data points
    for i, name in enumerate(target_names):
        mask = y == i
        fig_surf.add_trace(go.Scatter3d(
            x=X[mask, 2], y=X[mask, 3], z=X[mask, 0],
            mode="markers",
            marker=dict(size=5, color=colors[i], line=dict(width=1, color="black")),
            name=name,
        ))

    fig_surf.update_layout(
        height=550,
        scene=dict(
            xaxis_title="Petal length (cm)",
            yaxis_title="Petal width (cm)",
            zaxis_title="Sepal length (cm)",
        ),
        title="3D Decision Surface — drag to rotate",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    # 2D contour (static, as reference)
    st.subheader("2D Decision Contour (reference)")
    fig2d, ax2d = plt.subplots(figsize=(6, 4))
    ax2d.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    for i, name in enumerate(target_names):
        mask = y == i
        ax2d.scatter(X[mask, 2], X[mask, 3], c=colors[i], label=name, s=30, edgecolors="k")
    ax2d.set_xlabel("Petal length (cm)")
    ax2d.set_ylabel("Petal width (cm)")
    ax2d.set_title("2D Decision Contour")
    ax2d.legend()
    st.pyplot(fig2d)
    plt.close(fig2d)

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
