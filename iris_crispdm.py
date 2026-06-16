"""
Iris Classification using scikit-learn — CRISP-DM Methodology
==============================================================
CRISP-DM Phases:
  1. Business Understanding
  2. Data Understanding
  3. Data Preparation
  4. Modeling
  5. Evaluation
  6. Deployment
"""

# ============================================================
# 1. Business Understanding
# ============================================================
"""
Goal: Build a classifier that predicts Iris flower species
(setosa, versicolor, virginica) based on sepal/petal measurements.

Success Criteria: Achieve >=95% accuracy on held-out test data.
"""

# ============================================================
# 2. Data Understanding
# ============================================================
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 60)
print("PHASE 1 & 2: Business Understanding / Data Understanding")
print("=" * 60)
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Feature names: {feature_names}")
print(f"Target classes: {target_names}")
print(f"Class distribution: {[(n, (y == i).sum()) for i, n in enumerate(target_names)]}")
print()

# 3D scatter of raw data (Data Understanding)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["r", "g", "b"]
    for i, name in enumerate(target_names):
        mask = y == i
        ax.scatter(
            X[mask, 2],  # petal length
            X[mask, 3],  # petal width
            X[mask, 0],  # sepal length
            c=colors[i],
            label=name,
            s=40,
            edgecolors="k",
            alpha=0.8,
        )
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_zlabel("Sepal length (cm)")
    ax.set_title("Iris Dataset — 3D Scatter")
    ax.legend()
    fig.tight_layout()
    fig.savefig("iris_3d_scatter.png", dpi=150)
    plt.close(fig)
    print("3D scatter saved to iris_3d_scatter.png")
except ImportError:
    print("(matplotlib not available — skipping 3D scatter)")
print()

# ============================================================
# 3. Data Preparation
# ============================================================
print("=" * 60)
print("PHASE 3: Data Preparation")
print("=" * 60)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardise features (zero mean, unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Feature mean after scaling: {X_train_scaled.mean(axis=0).round(4)}")
print(f"Feature std  after scaling: {X_train_scaled.std(axis=0).round(4)}")
print()

# ============================================================
# 4. Modelling
# ============================================================
print("=" * 60)
print("PHASE 4: Modelling")
print("=" * 60)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# --- 4a. Candidate models with default hyper-parameters ---
models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=200),
    "RandomForest": RandomForestClassifier(random_state=42),
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"{name:20s} CV accuracy: {scores.mean():.4f} (±{scores.std():.4f})")

print()

# --- 4b. Hyper-parameter tuning on the best candidate (SVM) ---
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1, "scale", "auto"],
    "kernel": ["rbf"],
}

grid = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
print(f"Best SVM params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print()

# ============================================================
# 5. Evaluation
# ============================================================
print("=" * 60)
print("PHASE 5: Evaluation")
print("=" * 60)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

y_pred = best_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_acc:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print()

# Plot (optional, requires matplotlib)
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Iris Test Set")
    plt.savefig("iris_confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to iris_confusion_matrix.png")
except ImportError:
    print("(matplotlib not available — skipping plot)")
print()

# 3D decision boundary (requires matplotlib + trained scaler/model)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Use petal length (idx 2), petal width (idx 3) as x,y; fix sepal width at mean
    x_min, x_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    y_min, y_max = X[:, 3].min() - 0.5, X[:, 3].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))

    sepal_width_mean = X[:, 1].mean()
    sepal_length_mean = X[:, 0].mean()
    grid_4d = np.c_[np.full(xx.ravel().shape, sepal_length_mean),
                    np.full(xx.ravel().shape, sepal_width_mean),
                    xx.ravel(),
                    yy.ravel()]
    Z = best_model.predict(scaler.transform(grid_4d)).reshape(xx.shape)

    fig = plt.figure(figsize=(12, 5))

    # Left: 3D decision surface
    ax1 = fig.add_subplot(121, projection="3d")
    colors_arr = ["r", "g", "b"]
    Z_colors = np.zeros((*Z.shape, 3))
    for i, c in enumerate(colors_arr):
        Z_colors[Z == i] = plt.matplotlib.colors.to_rgb(c)
    zz = np.full_like(xx, sepal_length_mean)
    ax1.plot_surface(xx, yy, zz, facecolors=Z_colors,
                     alpha=0.4, rstride=1, cstride=1)

    # Overlay actual data points
    for i, name in enumerate(target_names):
        mask = y == i
        ax1.scatter(X[mask, 2], X[mask, 3], X[mask, 0],
                    c=colors_arr[i], label=name, s=30, edgecolors="k", alpha=0.8)

    ax1.set_xlabel("Petal length (cm)")
    ax1.set_ylabel("Petal width (cm)")
    ax1.set_zlabel("Sepal length (cm)")
    ax1.set_title("3D Decision Surface")
    ax1.legend()

    # Right: 2D decision contour (traditional view)
    ax2 = fig.add_subplot(122)
    ax2.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    for i, name in enumerate(target_names):
        mask = y == i
        ax2.scatter(X[mask, 2], X[mask, 3], c=colors_arr[i], label=name,
                    s=30, edgecolors="k")
    ax2.set_xlabel("Petal length (cm)")
    ax2.set_ylabel("Petal width (cm)")
    ax2.set_title("2D Decision Contour")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("iris_decision_boundary.png", dpi=150)
    plt.close(fig)
    print("Decision boundary saved to iris_decision_boundary.png")
except ImportError:
    print("(matplotlib not available — skipping decision boundary)")
print()

# Meet success criteria?
passed = test_acc >= 0.95
print(f"Success criterion (>=95% accuracy): {'PASSED' if passed else 'NOT PASSED'}")
print()

# ============================================================
# 6. Deployment
# ============================================================
print("=" * 60)
print("PHASE 6: Deployment")
print("=" * 60)


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """Predict Iris species from measurements.

    Parameters
    ----------
    sepal_length, sepal_width, petal_length, petal_width : float
        Iris flower measurements in cm.

    Returns
    -------
    species_name : str
        Predicted Iris species.
    """
    import numpy as np

    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    pred = best_model.predict(sample_scaled)
    return target_names[pred[0]]


# Quick smoke-test
demo_samples = [
    (5.1, 3.5, 1.4, 0.2),   # setosa
    (6.2, 2.9, 4.3, 1.3),   # versicolor
    (6.5, 3.0, 5.2, 2.0),   # virginica
]

print("Smoke-test predictions:")
for vals in demo_samples:
    species = predict_species(*vals)
    print(f"  {vals} -> {species}")

print()
print("=" * 60)
print("CRISP-DM pipeline complete.")
print("=" * 60)
