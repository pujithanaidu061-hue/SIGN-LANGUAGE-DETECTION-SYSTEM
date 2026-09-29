import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()

    if not success:
        print("Camera not working")
        break

    cv2.imshow("Camera Test", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()