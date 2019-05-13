import cv2
import numpy as np

kernel1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)

kernel2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 1]], np.uint8)

kernel3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)

kernel4 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0]], np.uint8)

file = 'healthy/15_h.jpg'
image = cv2.imread(file)

b, g, r = cv2.split(image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(g)
cv2.imwrite('Results/Clahe.jpg', enhanced)
tophat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)))
_, f6 = cv2.threshold(tophat, 5, 255, cv2.THRESH_OTSU)
cv2.imwrite('Results/Tophat1.jpg', tophat)
tophat2 = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel1)
cv2.imwrite('Results/Tophat2.jpg', tophat2)
tophat3 = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel2)
cv2.imwrite('Results/Tophat3.jpg', tophat3)
tophat4 = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel3)
cv2.imwrite('Results/Tophat4.jpg', tophat4)
tophat5 = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel4)
cv2.imwrite('Results/Tophat5.jpg', tophat5)
tophat_final = cv2.add(tophat, tophat2)
tophat_final = cv2.add(tophat_final, tophat3)
tophat_final = cv2.add(tophat_final, tophat4)
tophat_final = cv2.add(tophat_final, tophat5)
tophat_final = cv2.morphologyEx(tophat_final, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
cv2.imwrite('Results/TophatFinal.jpg', tophat_final)
lookUpTable = np.empty((1,256), np.uint8)
gamma = 1.0
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res = cv2.LUT(tophat_final, lookUpTable)
cv2.imwrite('Results/Res.jpg', res)


_, f6 = cv2.threshold(tophat, 5, 255, cv2.THRESH_OTSU)
cv2.imwrite('Results/Tresh.jpg', f6)
blur = cv2.medianBlur(f6, 9)
cv2.imwrite('Results/Blur.jpg', blur)
mask = np.ones(blur.shape[:2], dtype="uint8") * 255
contours, hierarchy = cv2.findContours(blur.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) <= 200:
        cv2.drawContours(mask, [cnt], -1, 0, -1)
im = cv2.bitwise_and(blur, blur, mask=mask)
cv2.imwrite('Results/Cont.jpg', im)


