import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors

# Wczytanie i przygotowanie danych
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
X = X.reshape(X.shape[0], -1) / 255.0

def cluster_accuracy(labels_true, labels_pred):
    clusters = set(labels_pred)
    clusters.discard(-1)
    total_correct = 0
    total_points = 0
    error_points = 0
    for cluster in clusters:
        indices = np.where(labels_pred == cluster)[0]
        true_labels_in_cluster = labels_true[indices]
        dominant_label = mode(true_labels_in_cluster, keepdims=True).mode[0]
        correct = np.sum(true_labels_in_cluster == dominant_label)
        total_correct += correct
        total_points += len(indices)
        error_points += len(indices) - correct
    accuracy = total_correct / total_points if total_points > 0 else 0
    noise_percent = np.sum(labels_pred == -1) / len(labels_pred) * 100
    error_percent = error_points / total_points * 100 if total_points > 0 else 0
    return accuracy, noise_percent, error_percent

def dbscan_manual(X, eps, min_samples):
    n = X.shape[0]
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    # Użycie NearestNeighbors do przyspieszenia
    nn = NearestNeighbors(radius=eps, n_jobs=-1)
    nn.fit(X)
    neighbors = nn.radius_neighbors(X, return_distance=False)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh = neighbors[i]
        if len(neigh) < min_samples:
            continue

        labels[i] = cluster_id
        seeds = list(neigh)
        seeds.remove(i)
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                neigh_j = neighbors[j]
                if len(neigh_j) >= min_samples:
                    for q in neigh_j:
                        if labels[q] == -1:
                            seeds.append(q)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels

pca_values = [10, 20, 30, 40, 50]
eps_values = np.linspace(1.5, 3.5, 9)
min_samples_values = [3, 5, 10]

best_result = None

print("Strojenie PCA, eps i min_samples (manual DBSCAN)...")

for seed in range(5):
    np.random.seed(seed)
    idx = np.random.choice(len(X), size=10000, replace=False)
    X_sample = X[idx]
    y_sample = y[idx]

    for pca_dim in pca_values:
        pca = PCA(n_components=pca_dim)
        X_pca = pca.fit_transform(X_sample)

        for eps in eps_values:
            for min_samples in min_samples_values:
                labels = dbscan_manual(X_pca, eps, min_samples)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 10 or n_clusters > 30:
                    continue
                accuracy, noise, errors = cluster_accuracy(y_sample, labels)
                print(f"Seed={seed} | PCA={pca_dim}, eps={eps:.2f}, min_samples={min_samples} | "
                      f"klastry={n_clusters}, dokładność={accuracy*100:.2f}%, "
                      f"szum={noise:.2f}%, błędy={errors:.2f}%")
                if best_result is None or (accuracy > best_result['accuracy'] and noise < best_result['noise_percent']):
                    best_result = {
                        'seed': seed,
                        'pca_dim': pca_dim,
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'accuracy': accuracy,
                        'noise_percent': noise,
                        'error_percent': errors,
                        'labels': labels
                    }

if best_result:
    print("\nNajlepszy wynik:")
    print(f"Seed: {best_result['seed']}")
    print(f"PCA: {best_result['pca_dim']}")
    print(f"eps: {best_result['eps']}")
    print(f"min_samples: {best_result['min_samples']}")
    print(f"Liczba klastrów: {best_result['n_clusters']}")
    print(f"Dokładność: {best_result['accuracy']*100:.2f}%")
    print(f"Procent szumu: {best_result['noise_percent']:.2f}%")
    print(f"Procent błędów: {best_result['error_percent']:.2f}%")
else:
    print("Nie znaleziono parametrów spełniających kryteria.")
