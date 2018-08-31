import cv2
import numpy as np
import math
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("-i", help="image location", default='images/sudoku.png')
parser.add_argument("-s", help="Show_image_processing", default=True)
args = parser.parse_args()
image_name = args.i
Display = str2bool(args.s)
IMAGE_WIDHT = 16
IMAGE_HEIGHT = 16
SUDOKU_SIZE= 9
N_MIN_ACTVE_PIXELS = 10


def show_images(name, img):
    if Display:
        cv2.imshow(name, img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def getOuterPoints(rcCorners):
    ar = []
    ar.append(rcCorners[0, 0, :])
    ar.append(rcCorners[1, 0, :])
    ar.append(rcCorners[2, 0, :])
    ar.append(rcCorners[3, 0, :])

    x_sum = sum(rcCorners[x, 0, 0] for x in range(len(rcCorners))) / len(rcCorners)
    y_sum = sum(rcCorners[x, 0, 1] for x in range(len(rcCorners))) / len(rcCorners)

    def algo(v):
        return (math.atan2(v[0] - x_sum, v[1] - y_sum)+ 2 * math.pi) % 2 * math.pi

    ar.sort(key=algo)

    return (ar[0], ar[3], ar[2], ar[1])


image_sudoku_original = cv2.imread(image_name)
show_images('first', image_sudoku_original)

gray = cv2.cvtColor(image_sudoku_original, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
show_images('bilateralFilter', gray)

thresh = cv2.adaptiveThreshold(gray,255, 1, 1, 11, 15)
show_images('threshold', thresh)

# applying canny edge detection doesn't work for finding the sudoku box
# edged = cv2.Canny(thresh, 30, 200)
# show_images('cannyEdge', edged)

__ , contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

rough_note = image_sudoku_original.copy()
cv2.drawContours(rough_note, contours, -1, (0, 255, 0), 3)
show_images('contours', rough_note)
#biggest rectangle
size_rectangle_max = 0
big_rectangle = None

for contour in contours:
    # aproximate countours to polygons
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # has the polygon 4 sides?
    if not (len(approximation) == 4):
        continue
    # is the polygon convex ?
    if not cv2.isContourConvex(approximation):
        continue
        # area of the polygon
    size_rectangle = cv2.contourArea(approximation)
    # store the biggest

    if size_rectangle > size_rectangle_max:
        size_rectangle_max = size_rectangle
        big_rectangle = approximation

rough_note = image_sudoku_original.copy()
cv2.drawContours(rough_note, [big_rectangle], 0, (0, 255, 0), 3)
show_images('Sudoku', rough_note)

points1 = np.array([
                    np.array([0.0, 0.0], np.float32) + np.array([144, 0], np.float32),
                    np.array([0.0, 0.0], np.float32),
                    np.array([0.0, 0.0], np.float32) + np.array([0.0, 144], np.float32),
                    np.array([0.0, 0.0], np.float32) + np.array([144, 144], np.float32),
                    ],np.float32)

outerPoints = getOuterPoints(big_rectangle)
points2 = np.array(outerPoints, np.float32)
pers = cv2.getPerspectiveTransform(points2,  points1)

warp = cv2.warpPerspective(image_sudoku_original, pers, (SUDOKU_SIZE*IMAGE_HEIGHT, SUDOKU_SIZE*IMAGE_WIDHT))
warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

show_images('transformed', warp_gray)



