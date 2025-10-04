import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from mha_toolbox.algorithms.pso import ParticleSwarmOptimization

data = load_breast_cancer()
X = data.data
y = data.target

# Normalize features for optimization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

def feature_selection_objective(solution):
    # Binary mask for selected features
    mask = (np.array(solution) > 0.5).astype(int)
    if np.sum(mask) == 0:
        return 1.0  # Penalize if no features selected
    # Simple classifier accuracy (KNN)
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    X_selected = X_scaled[:, mask == 1]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return 1.0 - acc  # Minimize error

pso = ParticleSwarmOptimization(population_size=20, max_iterations=30, verbose=False, mode=False)
result = pso.optimize(X=X_scaled, y=y, objective_function=feature_selection_objective)
result.plot_convergence()
result.save('results/test_breast_cancer_pso.json')
print('Best fitness:', result.best_fitness_)
print('Selected features:', result.n_selected_features_)
print('Selected indices:', result.selected_feature_indices_)
