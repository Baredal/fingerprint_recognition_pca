"""
PCA algorithm
"""
import numpy as np

class PCA:
    """
    PCA class
    """
    def __init__(self):
        self.selected_eigenvectors = np.array([])

    def power_iteration(self, covariance_matrix, num_components, iterations=200):
        """
        Power iteration to estimate the largest eigenvalues and eigenvectors
        """
        n = covariance_matrix.shape[0]
        eigenvectors = []

        for i in range(num_components):
            print("Component:", i)
            v = np.random.rand(n)
            for _ in range(iterations):
                v = np.dot(covariance_matrix, v)
                v /= np.linalg.norm(v)

            eigenvectors.append(v)

            # Deflate the covariance matrix to find the next eigenvector
            lambda_ = np.dot(v.T, np.dot(covariance_matrix, v))
            covariance_matrix -= lambda_ * np.outer(v, v)

        return np.array(eigenvectors).T

    def perform_pca(self, difference_matrix, num_components):
        """
        Perform PCA with power iteration
        """
        covariance_matrix = np.cov(difference_matrix, rowvar=True)
        print("Covariance matrix shape:", covariance_matrix.shape)

        eigenvectors = self.power_iteration(covariance_matrix, num_components)
        self.selected_eigenvectors = eigenvectors
        return eigenvectors

    def project_data(self, data_matrix, eigenvectors):
        """
        Project the data onto the selected eigenvectors
        """
        return np.dot(data_matrix.T, eigenvectors)
