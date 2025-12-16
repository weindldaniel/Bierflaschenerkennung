import cv2

cap = cv2.VideoCapture(1)   # HIER deine gefundene ID eintragen

if not cap.isOpened():
    print("Kamera kann nicht geöffnet werden.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Keine Videodaten – Kamera reagiert nicht.")
        break

    cv2.imshow("Kameratest", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()