import cv2

image = cv2.imread('shape.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    if i == 0:
        continue

    epsilon = 0.01*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    cv2.drawContours(image, contour, 0, (0,0,0), 4)

    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + (w/3))
    y_mid = int(y + (h/1.5))

    coords = (x_mid, y_mid)
    colour = (0,0,0)
    font = cv2.FONT_HERSHEY_DUPLEX

    if len(approx) == 3:
        cv2.putText(image, "Triangle", coords, font, 1, colour, 1) # Text on the image
    elif len(approx) == 4:
        cv2.putText(image, "Quadrilateral", coords, font, 1, colour, 1)
    elif len(approx) == 5:
        cv2.putText(image, "Pentagon", coords, font, 1, colour, 1)
    elif len(approx) == 6:
        cv2.putText(image, "Hexagon", coords, font, 1, colour, 1)
    else:
        # If the length is not any of the above, we will guess the shape/contour to be a circle.
        cv2.putText(image, "Circle", coords, font, 1, colour, 1)

cv2.imshow("shapes_detected", image)
cv2.waitKey(0)