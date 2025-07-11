import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
X = X.reshape(X.shape[0], -1)

def kmeans_analysis(n_clusters):
    print(f"\n--- Analiza dla {n_clusters} klastrów ---")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    print(f"Inercja dla {n_clusters} klastrów: {kmeans.inertia_:.2f}")
    
    # Macierz rozkładu cyfr
    confusion = np.zeros((n_clusters, 10))
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        digits_in_cluster = y[cluster_indices]
        for digit in range(10):
            confusion[cluster, digit] = np.sum(digits_in_cluster == digit)
    confusion_percent = confusion / confusion.sum(axis=1, keepdims=True) * 100
    
    # Wyświetlenie macierzy
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_percent, annot=True, fmt=".1f", cmap="Blues")
    plt.xlabel("Cyfry")
    plt.ylabel("Klastry")
    plt.title(f"Procentowy rozkład cyfr w {n_clusters} klastrach")
    plt.show()
    
    # Wyświetlenie centroidów
    cols = 5
    rows = (n_clusters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()
    for i in range(n_clusters):
        centroid_image = kmeans.cluster_centers_[i].reshape(28, 28)
        axes[i].imshow(centroid_image, cmap='gray')
        axes[i].set_title(f"Centroid {i}")
        axes[i].axis('off')
    for j in range(n_clusters, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f"Centroidy dla {n_clusters} klastrów")
    plt.show()
    
    return kmeans, confusion_percent

kmeans_10, confusion_10 = kmeans_analysis(10)
kmeans_15, confusion_15 = kmeans_analysis(15)
kmeans_20, confusion_20 = kmeans_analysis(20)
kmeans_30, confusion_30 = kmeans_analysis(30)