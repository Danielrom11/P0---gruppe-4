import cv2 as cv
import numpy as np
import os

# Main function containing the backbone of the program 222
def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")
    image_path = r"C:\Users\Jared\Downloads\King Domino train\73.jpg"
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
    if 12 < hue < 36 and 222 < saturation < 264 and 125 < value < 211:
        return "Field"
    if 18 < hue < 68 and 73 < saturation < 216 and 24 < value < 73:
        return "Forest"
    if 95 < hue < 119 and 230 < saturation < 264 and 107 < value < 194:
        return "Lake"
    if 25 < hue < 54 and 189 < saturation < 257 and 64 < value < 168:
        return "Grassland"
    if 8 < hue < 33 and 65 < saturation < 176 and 72 < value < 128:
        return "Swamp"
    if 9 < hue < 34 and 54 < saturation < 129 and 21 < value < 74:
        return "Mine"
    if 11 < hue < 97 and 31 < saturation < 204 and 49 < value < 130:
        return "Home"
    return "Unknown"

if __name__ == "__main__":
    main()