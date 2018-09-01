import numpy as np
import math
import cv2

Display = True
IMAGE_WIDHT = 16
IMAGE_HEIGHT = 16
SUDOKU_SIZE = 9
N_MIN_ACTVE_PIXELS = 10


def show_images(name, img):
    if Display:
        cv2.imshow(name, img)
        cv2.waitKey()
        cv2.destroyAllWindows()


warp_gray = cv2.imread('images/wrap_gray.jpg')
show_images('ip', warp_gray)


def extract_number(x, y):
    # square -> position x-y
    im_number = warp_gray[x*IMAGE_HEIGHT:(x+1)*IMAGE_HEIGHT][:, y*IMAGE_WIDHT:(y+1)*IMAGE_WIDHT]
    show_images('%i,%i' % (x, y), im_number)
    # threshold
    im_number_thresh = cv2.adaptiveThreshold(im_number, 255, 1, 1, 15, 9)
    # delete active pixel in a radius (from center)
    for i in range(im_number.shape[0]):
        for j in range(im_number.shape[1]):
            dist_center = math.sqrt((IMAGE_WIDHT/2 - i)**2 + (IMAGE_HEIGHT/2 - j)**2)
            if dist_center > 6:
                im_number_thresh[i,j] = 0

    n_active_pixels = cv2.countNonZero(im_number_thresh)
    return [im_number, im_number_thresh, n_active_pixels]


def find_biggest_bounding_box(im_number_thresh):
    contour, hierarchy = cv2.findContours(im_number_thresh.copy(),
                                          cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)

    biggest_bound_rect = []
    bound_rect_max_size = 0
    for i in range(len(contour)):
        bound_rect = cv2.boundingRect(contour[i])
        size_bound_rect = bound_rect[2] * bound_rect[3]
        if size_bound_rect > bound_rect_max_size:
            bound_rect_max_size = size_bound_rect
            biggest_bound_rect = bound_rect
    # bounding box a little more bigger
    x_b, y_b, w, h = biggest_bound_rect
    x_b = x_b - 1
    y_b = y_b - 1
    w = w + 2
    h = h + 2

    return [x_b, y_b, w, h]


def recognize_number(x, y):
    """
    Recognize the number in the rectangle
    """
    # extract the number (small squares)
    [im_number, im_number_thresh, n_active_pixels] = extract_number(x, y)

    if n_active_pixels > N_MIN_ACTVE_PIXELS:
        [x_b, y_b, w, h] = find_biggest_bounding_box(im_number_thresh)

        im_t = cv2.adaptiveThreshold(im_number, 255, 1, 1, 15, 9)
        number = im_t[y_b:y_b+h, x_b:x_b+w]

        if number.shape[0]*number.shape[1] > 0:
            number = cv2.resize(number, (IMAGE_WIDHT, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            ret, number2 = cv2.threshold(number, 127, 255, 0)
            number = number2.reshape(1, IMAGE_WIDHT*IMAGE_HEIGHT)
            sudoku[x*9+y, :] = number
            return 1

        else:
            sudoku[x*9+y, :] = np.zeros(shape=(1, IMAGE_WIDHT*IMAGE_HEIGHT))
            return 0


sudoku = np.zeros(shape=(9*9, IMAGE_WIDHT*IMAGE_HEIGHT))
n_numbers = 0
indexes_numbers = []
for i in range(SUDOKU_SIZE):
    for j in range(SUDOKU_SIZE):
        if recognize_number(i, j) == 1:
            indexes_numbers.insert(n_numbers, i*9+j)
            n_numbers = n_numbers+1

print(sudoku)
