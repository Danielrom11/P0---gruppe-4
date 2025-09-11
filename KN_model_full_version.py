import cv2 as cv
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# --- Load KNN model and label encoder ---
# Load your training data (same as in knn_model_til_kd.py)
df = pd.read_csv(r'C:\Users\danie\Desktop\python_work\P0---gruppe-4\data_KD.csv', sep=';')
df['h'] = pd.to_numeric(df['h'], errors='coerce')
df['s'] = pd.to_numeric(df['s'], errors='coerce')
df['v'] = pd.to_numeric(df['v'], errors='coerce')
df.dropna(subset=['h', 's', 'v', 'target'], inplace=True)
X = df[['h', 's', 'v']].values
y = df['target'].values

# Fit KNN model
k = 11
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X, y)

# --- Main tile labeling code ---
def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")
    image_path = r"C:\Users\danie\Downloads\King Domino dataset\71.jpg"
    if not os.path.isfile(image_path):
        print("Image not found")
        return
    image = cv.imread(image_path)
    tiles = get_tiles(image)
    print(len(tiles))
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            print(f"Tile ({x}, {y}):")
            print(get_terrain_knn(tile))
            print("=====")

def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

# Use KNN to determine terrain type
def get_terrain_knn(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0,1))
    features = np.array([[hue, saturation, value]])
    print(f"Hue: {hue}, Saturation: {saturation}, Value: {value}")
    prediction = knn_classifier.predict(features)
    return prediction[0]

if __name__ == "__main__":
    main()