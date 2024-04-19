import numpy as np
import cv2
import math
from PIL import Image

frequencies = {}
gcd_values = set()
thresholds = {}

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def get_gcds(intensities):
    gcds = set()
    for i in range(len(intensities)):
        if i == len(intensities)-1:
            continue
        x, y = intensities[i], intensities[i+1]
        i+=1
        g = math.gcd(x, y)
        if g != 1:
            gcds.add(g)
    print(np.mean(list(gcds)))
    return gcds

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
    padded_array = np.pad(gray_image.astype(np.float64),((1, 1), (1, 1)), mode='edge')

    smoothed_image = np.zeros_like(img_array, dtype=np.float64)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            window = padded_array[i:i + 3, j:j + 3]
            smoothed_image[i, j] = np.sum(window * filter)

    smoothed_image2 = np.clip(smoothed_image, 0, 255).astype(np.uint8)

    smoothed_image_2_pil = Image.fromarray(smoothed_image2)
    smoothed_image_2_pil.save(f'Smoothed_2_{image_path}')


def gcd_threshold_segmentation(image_path):
    global thresholds
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    horizontal = []
    vertical = []
    spiral = []
    rows = len(gray_image)
    cols = len(gray_image[0])
    unique_values = np.unique(gray_image.flatten())
    top_row, bottom_row = 0, rows - 1
    left_col, right_col = 0, cols - 1


    #ALL PAIRS
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


    #ITERATE HORIZONTALLY
    for row in range(rows):
        for col in range(cols):
            horizontal.append(gray_image[row][col])
    # horizontal = np.unique(horizontal)

    #ITERATE VERTICALLY
    for col in range(cols):
        for row in range(rows):
            vertical.append(gray_image[row][col])
    # vertical = np.unique(vertical)

    #ITERATE IN SPIRAL
    while top_row <= bottom_row and left_col <= right_col:
        for col in range(left_col, right_col + 1):
            spiral.append(gray_image[top_row][col])
        top_row += 1

        for row in range(top_row, bottom_row + 1):
            spiral.append(gray_image[row][right_col])
        right_col -= 1

        if top_row <= bottom_row:
            for col in range(right_col, left_col - 1, -1):
                spiral.append(gray_image[bottom_row][col])
            bottom_row -= 1

        if left_col <= right_col:
            for row in range(bottom_row, top_row - 1, -1):
                spiral.append(gray_image[row][left_col])
            left_col += 1
    # spiral = np.unique(spiral)


    #ITERATE DIAGONAL
    if rows<cols:
        main_diagonal = [gray_image[i][i] for i in range(rows)]
        anti_diagonal = [gray_image[i][rows-1-i] for i in range(rows)]
    else:
        main_diagonal = [gray_image[i][i] for i in range(cols)]
        anti_diagonal = [gray_image[i][cols-1-i] for i in range(cols)]

    horizontal_gcds = get_gcds(horizontal)
    vertical_gcds = get_gcds(vertical)
    spiral_gcds = get_gcds(spiral)
    diagonal_gcds = get_gcds(main_diagonal+anti_diagonal)

    mean_allvals = round(np.mean(list(gcd_values)),2)

    mean_horizontal = round(np.mean(list(horizontal_gcds)),2)
    median_horizontal = round(np.median(list(horizontal_gcds)),2)

    mean_vertical = round(np.mean(list(vertical_gcds)),2)
    median_vertical = round(np.median(list(vertical_gcds)),2)

    mean_spiral = round(np.mean(list(spiral_gcds)),2)
    median_spiral = round(np.median(list(spiral_gcds)),2)

    mean_diagonal = round(np.mean(list(diagonal_gcds)),2)
    median_diagonal = round(np.median(list(diagonal_gcds)),2)


    thresholds['allvalues'] = mean_allvals
    thresholds['horizontal'] = mean_horizontal
    # thresholds['Horizontal_median'] = median_horizontal
    thresholds['vertical'] = mean_vertical
    # thresholds['vertical_median'] = median_vertical
    thresholds['spiral'] = mean_spiral
    # thresholds['spiral_median'] = median_spiral
    thresholds['diagonal'] = mean_diagonal
    # thresholds['diagonal_median'] = median_diagonal

    return thresholds

def segmentation(image_path,type,T):
    gimage = cv2.imread(image_path,0)
    m, n = gimage.shape
    img_thresh = np.zeros((m, n), dtype=int)

    for i in range(m):
        for j in range(n):
            if gimage[i, j] < T:
                img_thresh[i, j] = 0;
            else:
                img_thresh[i, j] = 255;

    cv2.imwrite(f'{type}_segmented_{image_path}',img_thresh)

if __name__ == "__main__":
    image_list = ["mark1.jpg"]

    for x in image_list:
        image_path = f"images/{x}"

        threshold_values = gcd_threshold_segmentation(image_path)
        for type,threshold_value in threshold_values.items():
            segmentation(image_path,type,int(threshold_value))


        #Divide image into 4 parts find gcds in each of the 4 areas and segment each part respectively and then add all images compare results

        # filter1 = find_filter1()
        # filter2 = find_filter2()
        # smooth1(image_path,filter1)
        # smooth2(image_path,filter2)