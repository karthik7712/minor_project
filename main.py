import numpy as np
import cv2
import math
from PIL import Image

frequencies = {}
gcd_values = set()


def find_gcd_vectorized(arr):
    gcd_vectorized = np.frompyfunc(math.gcd, 2, 1)
    return gcd_vectorized.reduce(arr, axis=0)


def segmentation(image_path,T):
    gimage = cv2.imread(image_path,0)
    m, n = gimage.shape
    img_thresh = np.zeros((m, n), dtype=int)

    for i in range(m):
        for j in range(n):
            if gimage[i, j] < T:
                img_thresh[i, j] = 0;
            else:
                img_thresh[i, j] = 255;

    cv2.imwrite(f'segmented_{image_path}',img_thresh)


def gcd_threshold_segmentation(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    unique_values = np.unique(gray_image.flatten())

    for i in range(len(unique_values)):
        for j in range(i + 1, len(unique_values)):
            x, y = unique_values[i], unique_values[j]
            gcd = math.gcd(x, y)
            if gcd in frequencies:
                frequencies[gcd] += 1
            else:
                frequencies[gcd] = 1
            if gcd != 1:
                gcd_values.add(gcd)

    print(round(np.mean(list(gcd_values))),2)
    print(np.median(list(gcd_values)))
    print(round(np.std(list(gcd_values))),2)
    return np.mean(list(gcd_values))


if __name__ == "__main__":
    image_list = ["apples.jpg","BT_orig.png","logo.png"]

    for x in image_list:
        image_path = f"images/{x}"

        threshold_value = gcd_threshold_segmentation(image_path)
        segmentation(image_path,threshold_value)

