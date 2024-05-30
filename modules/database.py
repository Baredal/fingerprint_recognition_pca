import os
import numpy as np
from PCAnalyzer import PCA
from finger import Finger

class Database:
    """
    DBS class
    """
    def __init__(self) -> None:
        self.data_fingers = []
        self.data_matrix = np.array([])
        self.average_matrix = np.array([])
        self.pca = PCA()

    def load_fingers(self, data_folder, subfolder_name):
        """
        Loads fingers from 1 specified subfolder into the database into data_fingers attribute
        """
        print("Loading images...")
        finger_list = []
        subfolder_path = os.path.join(data_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(file_path):
                    finger = Finger(file_path)
                    finger_list.append(finger)
            self.data_fingers = finger_list
            print("Loaded succesfully.")
        else:
            raise FileNotFoundError(f"Subfolder '{subfolder_name}' not found.")

    def load_all_fingers(self, data_folder="train_data"):
        """
        Loads all fingers from all subfolders into the database into data_fingers attribute
        """
        print("Loading images...")
        finger_list = []
        for i in range(1,51):
            subfolder_name = f'sub{i}'
            subfolder_path = os.path.join(data_folder, subfolder_name)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, filename)
                    if os.path.isfile(file_path):
                        finger = Finger(file_path)
                        finger_list.append(finger)
                self.data_fingers = finger_list
                print("Loaded succesfully.")
            else:
                raise FileNotFoundError(f"Subfolder '{subfolder_name}' not found.")

    def fingers_to_matrix(self):
        """
        Takes data_fingers and transforms them into a flattened vector
        consisting of pixel values (0-255)
        """
        if len(self.data_fingers) == 0:
            raise ValueError("No fingers loaded.")
        print("Adding images to matrix...")
        flattened_images = []
        for finger in self.data_fingers:
            flattened_image = np.array(finger.image).flatten()
            flattened_images.append(flattened_image)

        self.data_matrix = np.array(flattened_images).T
        print("Added succesfully.")
        return self.data_matrix

    def compute_average_image(self):
        """
        Compute the average image
        """
        row_mean = np.mean(self.data_matrix, axis=1)
        row_mean_matrix = np.tile(row_mean[:, None], (1, self.data_matrix.shape[1]))

        self.average_matrix = self.data_matrix - row_mean_matrix
        return self.average_matrix

    def perform_pca(self, num_components):
        """
        Perform PCA on the average matrix
        """
        return self.pca.perform_pca(self.average_matrix, num_components)