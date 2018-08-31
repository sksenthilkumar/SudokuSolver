import cv2

Display = True

image_sudoku_original = cv2.imread('download.png')
if Display:
    cv2.imshow('image_sudoku_original', image_sudoku_original)


#gray image
image_sudoku_gray = cv2.cvtColor(image_sudoku_original,cv2.COLOR_BGR2GRAY)
cv2.imshow('image_sudoku_gray', image_sudoku_gray)
#adaptive threshold
thresh = cv2.adaptiveThreshold(image_sudoku_gray,255,1,1,15,1)

if Display:
    cv2.imshow('thresh', thresh)

__ , contours0,hierarchy = cv2.findContours( thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


#copy the original image to show the posible candidate
image_sudoku_candidates = image_sudoku_original.copy()


#biggest rectangle
size_rectangle_max = 0
big_rectangle = None
for i in range(len(contours0)):
    # aproximate countours to polygons

    approximation = cv2.approxPolyDP(contours0[i], 4, True)


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
#show the best candidate
approximation = big_rectangle
for i in range(len(approximation)):
    cv2.line(image_sudoku_candidates,
             (big_rectangle[(i%4)][0][0], big_rectangle[(i%4)][0][1]),
             (big_rectangle[((i+1)%4)][0][0], big_rectangle[((i+1)%4)][0][1]),
             (255, 0, 0), 2)



