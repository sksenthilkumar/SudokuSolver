import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", help="image location", default= 'images/sudoku.png')
parser.add_argument("-s", help="Show_image_processing", default=True)
args = parser.parse_args()
image_name = args.i
Display = args.s


def show_images(name, img):
    if Display:
        cv2.imshow(name, img)

    cv2.waitKey()
    cv2.destroyAllWindows()


image_sudoku_original = cv2.imread(image_name)
show_images('first', image_sudoku_original)

gray = cv2.cvtColor(image_sudoku_original, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
show_images('bilateralFilter', gray)

thresh = cv2.adaptiveThreshold(gray,255,1,1,11,15)
show_images('threshold', thresh)

# applying canny edge detection doesn't work for finding the sudoku box
#edged = cv2.Canny(thresh, 30, 200)
#show_images('cannyEdge', edged)

__ , contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
print(len(contours))

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


