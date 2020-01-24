import cv2


img = cv2.imread("../../img/cameraman.tif", cv2.IMREAD_GRAYSCALE)

gauimg = cv2.GaussianBlur(img, (0, 0), cv2.BORDER_DEFAULT)


cv2.imshow("adfasdf", img-gauimg)
cv2.waitKey(0)

cv2.destroyAllWindows()
