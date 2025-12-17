import cv2
import time

def open_camera_full_reset(cam_id=0):
    # 1️⃣ Alte Handles sicher schließen
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(1.0)  # <<< extrem wichtig!

    # 2️⃣ Kamera neu öffnen
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Kamera konnte nicht geöffnet werden")

    # 3️⃣ ALLES explizit setzen (erzwingt Neuinitialisierung)
    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Optional: Belichtung / Auto-Modi
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = auto (DSHOW)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    # 4️⃣ Dummy-Reads → leere Puffer verwerfen
    for _ in range(10):
        cap.read()

    return cap


#%%
cap = open_camera_full_reset(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
