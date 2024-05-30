from database import Database
from finger import Finger

import numpy as np

if __name__ == "__main__":
    db = Database()
    data_path = 'data' #will take 150x150 images, change to data100 or initial_data for 100x100 and 200x200 sizes respectively
    subfold_name = 'sub1' #loads images from sub1 folder, enter the one u want or load_all_fingers
    db.load_fingers(data_path, subfold_name)

    #uncomment to load all images from all subfolders, in this case dont forget to comment 2 lines above
    # db.load_all_fingers()
    
    db.fingers_to_matrix()
    db.compute_average_image()

    num_components = 100 #change the value if needed
    principal_components = db.perform_pca(num_components)

    projected_data = db.pca.project_data(db.data_matrix, principal_components)
    projected_data = projected_data.T
    print(projected_data.shape)

    path = input("Please input path for the photo >>> ")

    finger = np.array(Finger(path).image).flatten()
    projected_finger = db.pca.project_data(finger, principal_components)

    distances = []
    for i in range(projected_data.shape[1]):
        distance = np.linalg.norm(projected_data[:, i] - projected_finger)
        distances.append(distance)

    print("Euclidean distances with projected_finger:")
    for index, distance in enumerate(distances):
        print(f"Distance with finger {index}: {distance}")

    min_distance = min(distances)
    print(min_distance, distances.index(min_distance))
    threshold = 2000
    is_in_dataset = min_distance < threshold

    print("Is the projected image from the dataset?", is_in_dataset)
