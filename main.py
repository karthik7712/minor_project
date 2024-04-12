import numpy as np
import cv2
import math
from PIL import Image

frequencies = {}
gcd_values = set()


def select_numbers_with_least_variation(numbers, k):
    numbers.sort()
    n = len(numbers)
    interval = n // (k + 1)

    selected_numbers = []
    for i in range(1, k + 1):
        index = i * interval
        selected_numbers.append(numbers[index])

    return selected_numbers



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


def find_filter1():
    global frequencies
    top_gcd = []
    count = 0
    for key in frequencies.keys():
        if count<9:
            top_gcd.append(key)
            count += 1
        else:
            break

    matrix = [top_gcd[i:i + 3] for i in range(0, len(top_gcd), 3)]
    # print(matrix)

    vals = np.array(matrix)
    x = np.sum(vals)
    filter_3x3 = vals / x
    print("The filter is :", filter_3x3)
    return filter_3x3


def smooth1(image_path,filter):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray_image)
    filtered_image = np.zeros_like(img_array)

    for y in range(1, img_array.shape[0] - 1):
        for x in range(1, img_array.shape[1] - 1):
            filtered_image[y, x] = round(np.sum(img_array[y - 1:y + 2, x - 1:x + 2] * filter), 0)

    filtered_image = np.clip(filtered_image, 0, 255)

    smoothed_image = Image.fromarray(filtered_image.astype(np.uint8))
    smoothed_image.save(f"smoothed_1_{image_path}")


def find_filter2():
    global gcd_values
    top_gcd2 = select_numbers_with_least_variation(list(gcd_values),9)
    filter_matrix2 = np.array(top_gcd2).reshape(3, 3) / np.sum(top_gcd2)
    return filter_matrix2


def smooth2(image_path,filter):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray_image)
    padded_array = np.pad(gray_image.astype(np.float64), ((1, 1), (1, 1)), mode='edge')

    smoothed_image = np.zeros_like(img_array, dtype=np.float64)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            window = padded_array[i:i + 3, j:j + 3]
            smoothed_image[i, j] = np.sum(window * filter)

    smoothed_image2 = np.clip(smoothed_image, 0, 255).astype(np.uint8)

    smoothed_image_2_pil = Image.fromarray(smoothed_image2)
    smoothed_image_2_pil.save(f'Smoothed_2_{image_path}')


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
    image_list = ["apples.jpg","BT_orig.png","logo.png","phone (5).jpg"]

    for x in image_list:
        image_path = f"images/{x}"

        threshold_value = gcd_threshold_segmentation(image_path)
        segmentation(image_path,threshold_value)
        filter1 = find_filter1()
        filter2 = find_filter2()
        smooth1(image_path,filter1)
        smooth2(image_path,filter2)
