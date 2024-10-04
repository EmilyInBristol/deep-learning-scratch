import numpy as np
import random

# Step 1: Generate synthetic data for testing
def generate_data(n_samples, n_features):
    # Generate random vectors in a high-dimensional space
    data = np.random.randn(n_samples, n_features) * 100
    return data

# Step 2: Split vectors into m subspaces (subvectors)
def split_vector(vector, m):
    # Split the vector into m equal-length parts (subspaces)
    subvector_len = len(vector) // m
    return [vector[i * subvector_len:(i + 1) * subvector_len] for i in range(m)]

# Step 3: K-means implementation for quantizing subvectors
def kmeans_subvector(subvectors, k, max_iters=100, tolerance=1e-4):
    # Randomly initialize k centroids from the subvectors
    centroids = subvectors[random.sample(range(len(subvectors)), k)]
    
    for _ in range(max_iters):
        # Assign each subvector to the nearest centroid
        clusters = assign_clusters(subvectors, centroids)
        
        # Update centroids based on the mean of assigned subvectors
        new_centroids = update_centroids(subvectors, clusters, k)
        
        # Check for convergence (centroid shifts)
        centroid_shifts = np.linalg.norm(centroids - new_centroids, axis=1)
        if np.all(centroid_shifts < tolerance):
            break
        
        centroids = new_centroids
    
    return centroids, clusters

# Assign subvectors to the nearest centroid
def assign_clusters(subvectors, centroids):
    clusters = []
    for subvector in subvectors:
        distances = [np.linalg.norm(subvector - centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)

# Update the centroids based on assigned subvectors
def update_centroids(subvectors, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = np.array([subvectors[j] for j in range(len(clusters)) if clusters[j] == i])
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            # Random reinitialization if a cluster has no points
            new_centroid = subvectors[random.randint(0, len(subvectors) - 1)]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# Step 4: Product Quantization process
def product_quantization(X, m, k):
    n_samples, n_features = X.shape
    
    # Ensure that n_features is divisible by m
    assert n_features % m == 0, "Number of features must be divisible by m"
    
    subvector_len = n_features // m
    
    # Codebooks for each subspace
    codebooks = []
    
    # For each subspace, train k-means to build codebooks
    for i in range(m):
        # Extract the i-th subspace (subvector)
        subvectors = X[:, i * subvector_len:(i + 1) * subvector_len]
        
        # Train k-means on this subspace
        centroids, clusters = kmeans_subvector(subvectors, k)
        
        # Store the codebook (centroids)
        codebooks.append(centroids)
    
    return codebooks

# Step 5: Encode the vectors using the trained codebooks
def encode_vectors(X, codebooks):
    n_samples, n_features = X.shape
    m = len(codebooks)
    subvector_len = n_features // m
    
    # Encoding: replace each subvector by the nearest codeword from the codebook
    encoded_vectors = []
    
    for i in range(n_samples):
        encoded = []
        for j in range(m):
            subvector = X[i, j * subvector_len:(j + 1) * subvector_len]
            distances = [np.linalg.norm(subvector - codeword) for codeword in codebooks[j]]
            encoded.append(np.argmin(distances))  # Store the index of the nearest centroid
        encoded_vectors.append(encoded)
    
    return np.array(encoded_vectors)

# Step 6: Reconstruct the vectors from the encoded representation
def reconstruct_vectors(encoded_vectors, codebooks):
    n_samples, m = encoded_vectors.shape
    subvector_len = codebooks[0].shape[1]
    
    # Reconstruct each vector by replacing each index with the corresponding centroid
    reconstructed_vectors = []
    
    for i in range(n_samples):
        reconstructed = []
        for j in range(m):
            centroid_index = encoded_vectors[i, j]
            reconstructed.append(codebooks[j][centroid_index])
        reconstructed_vectors.append(np.hstack(reconstructed))
    
    return np.array(reconstructed_vectors)

# Main function to run PQ
if __name__ == "__main__":
    # Step 1: Generate synthetic data (1000 samples, 8-dimensional vectors)
    X = generate_data(n_samples=1000, n_features=8)
    
    # Number of subspaces (m) and codebook size (k)
    m = 4  # Divide 8-dimensional vectors into 4 subvectors (each subvector will be 2D)
    k = 16  # 16 centroids in each subspace (codebook size)
    
    # Step 4: Train Product Quantization (PQ)
    codebooks = product_quantization(X, m, k)
    print(len(codebooks), len(codebooks[0]))
    
    # Step 5: Encode the vectors using the trained codebooks
    encoded_vectors = encode_vectors(X, codebooks)
    
    # Step 6: Reconstruct the vectors from the encoded representation
    reconstructed_X = reconstruct_vectors(encoded_vectors, codebooks)
    
    # Show the difference between original and reconstructed vectors
    print("Original Vectors:\n", X[:5])  # Print first 5 original vectors
    print("\nReconstructed Vectors:\n", reconstructed_X[:5])  # Print first 5 reconstructed vectors
    
    