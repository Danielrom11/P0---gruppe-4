import cv2 as cv
import numpy as np
import os

# Main function containing the backbone of the program
def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")
    image_path = r"C:\Users\danie\Downloads\King Domino dataset\40.jpg" # der er brugt jpg nr. 1, 13, 45, 65, 22 og 42 til test af hsv
    if not os.path.isfile(image_path):
        print("Image not found")
        return
    image = cv.imread(image_path)
    tiles = get_tiles(image)
    print(len(tiles))
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            print(f"Tile ({x}, {y}):")
            print(get_terrain(tile))
            print("=====")

# Break a board into tiles
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

# Determine the type of terrain in a tile
def get_terrain(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0,1)) # Consider using median instead of mean
    print(f"H: {hue}, S: {saturation}, V: {value}")
    if 5 < hue < 45 and 210 < saturation < 275 and 150 < value < 230:
        return "Field"
    if 30 < hue < 60 and 70 < saturation < 205 and 30 < value < 75:
        return "Forest"
    if 85 < hue < 135 and 200 < saturation < 275 and 125 < value < 190:
        return "Lake"
    if 30 < hue < 55 and 150 < saturation < 250 and 110 < value < 165:
        return "Grassland"
    if 5 < hue < 45 and 35 < saturation < 155 and 75 < value < 145:
        return "Swamp"
    if 15 < hue < 35 and 55 < saturation < 130 and 25 < value < 60:
        return "Mine"
    if 25 < hue < 110 and 35 < saturation < 130 and 40 < value < 100:
        return "Home"
    return "Unknown"

if __name__ == "__main__":
    main()


    # hvad er bedst at bruge median eller mean?
    # bedste måde at træne den på?