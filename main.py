import cv2
import numpy as np
cap = cv2.VideoCapture('aa.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
min_contour_area = 1400
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    column_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            center = (int((2 * x + w) / 2), int((2 * y + h) / 2))
            print(center)
            column_count += 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            center = (int((2 * x + w) / 2), int((2 * y + h) / 2))
            if column_count > 5:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            elif column_count > 2:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, 'Count:  '+str(column_count), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Traffic Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
