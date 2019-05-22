import cv2
import numpy as np


healthyFolder = 'Resized/healthy/'
files = ['Resized11_h', 'Resized12_h', 'Resized13_h', 'Resized14_h', 'Resized15_h']

for file in files:
    image = cv2.imread(healthyFolder + file + '.jpg')
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(g)
    cv2.imwrite('results/' + file + '/Clahe.jpg', enhanced)
    image_enhanced = cv2.equalizeHist(enhanced)
    cv2.imwrite('results/' + file + '/Equalized.jpg', image_enhanced)
    tophat1 = cv2.morphologyEx(image_enhanced, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    tophat2 = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27)))
    cv2.imwrite('results/' + file + '/Tophat1.jpg', tophat1)
    cv2.imwrite('results/' + file + '/Tophat2.jpg', tophat2)
    dd1 = clahe.apply(tophat1)
    dd2 = clahe.apply(tophat2)
    cv2.imwrite('results/' + file + '/dd1.jpg', dd1)
    cv2.imwrite('results/' + file + '/dd2.jpg', dd2)
    blur1 = cv2.morphologyEx(tophat1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    blur2 = cv2.morphologyEx(tophat2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cv2.imwrite('results/' + file + '/Open1.jpg', blur1)
    cv2.imwrite('results/' + file + '/Open2.jpg', blur2)
    blur1 = cv2.morphologyEx(blur1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    blur2 = cv2.morphologyEx(blur2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cv2.imwrite('results/' + file + '/Close1.jpg', blur1)
    cv2.imwrite('results/' + file + '/Close2.jpg', blur2)
    _, t1 = cv2.threshold(dd1, 110, 255, cv2.THRESH_BINARY)
    _, t2 = cv2.threshold(dd2, 60, 255, cv2.THRESH_BINARY)
    cv2.imwrite('results/' + file + '/Tresh1.jpg', t1)
    cv2.imwrite('results/' + file + '/Tresh2.jpg', t2)
    blur1 = cv2.medianBlur(t1, 5)
    blur2 = cv2.medianBlur(t2, 5)
    cv2.imwrite('results/' + file + '/Blur1.jpg', blur1)
    cv2.imwrite('results/' + file + '/Blur2.jpg', blur2)
    mask1 = np.ones(blur1.shape[:2], dtype="uint8") * 255
    contours1, hierarchy1 = cv2.findContours(blur1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours1:
        if cv2.contourArea(cnt) <= 150:
            cv2.drawContours(mask1, [cnt], -1, 0, -1)
    im1 = cv2.bitwise_and(blur1, blur1, mask=mask1)
    cv2.imwrite('results/' + file + '/Cont1.jpg', im1)
    mask2 = np.ones(blur2.shape[:2], dtype="uint8") * 255
    contours2, hierarchy2 = cv2.findContours(blur2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        if cv2.contourArea(cnt) <= 150:
            cv2.drawContours(mask2, [cnt], -1, 0, -1)
    im2 = cv2.bitwise_and(blur2, blur2, mask=mask2)
    cv2.imwrite('results/' + file + '/Cont2.jpg', im2)
    final1 = cv2.add(im1, im2)
    cv2.imwrite('results/' + file + '/' + file + 'Final1.jpg', final1)



