# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Bierflaschenerkennung mit Live-Adjustment
# ------------------------------------------------------------

import cv2
import numpy as np
import time

# ------------------------------------------------------------
# 1. KAMERA FULL RESET
# ------------------------------------------------------------
def open_camera_full_reset(cam_id=0):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(1.0)

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Kamera konnte nicht geöffnet werden")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(10):
        cap.read()
    return cap

# ------------------------------------------------------------
# 2. HELLIGKEIT / KONTRAST / GAMMA
# ------------------------------------------------------------
def adjust_brightness_contrast_gamma(img, brightness=247, contrast=89, gamma=0.7):
    # Falls Graubild → BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    alpha = 1 + (contrast / 100.0)
    beta = brightness
    img_lin = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    img_float = img_lin.astype(np.float32) / 255.0
    gamma_corrected = np.power(img_float, 1.0 / gamma)
    gamma_corrected = np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)
    return gamma_corrected

# ------------------------------------------------------------
# 3. FARBE / GRÜNKANAL
# ------------------------------------------------------------
def extract_green_channel(img):
    return img[:, :, 1]

# ------------------------------------------------------------
# 3. Treshold
# ------------------------------------------------------------
def apply_threshold(gray_img, thresh_value=135):
    """
    Threshold-Anpassung wie in NI Vision Assistant
    Wertebereich: 0 - 255
    """
    thresh_value = np.clip(thresh_value, 0, 255)  # Sicherheit
    _, binary = cv2.threshold(gray_img, thresh_value, 255, cv2.THRESH_BINARY)
    return binary

# ------------------------------------------------------------
# 5. AREA FILTER
# ------------------------------------------------------------
def filter_by_area(binary_img, min_area=7000, max_area=18000):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(binary_img)
    
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            cv2.drawContours(output, [c], -1, 255, -1)
    
    return output

# ------------------------------------------------------------
# 6. ELONGATION FACTOR FILTER
# ------------------------------------------------------------
def filter_by_elongation(binary_img, min_elong=0, max_elong=3):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(binary_img)
    
    for c in contours:
        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            (center, axes, angle) = ellipse
            major = max(axes)
            minor = min(axes)
            elongation = major / minor if minor > 0 else 999
        else:
            elongation = 999  # zu kleine Konturen ignorieren

        if min_elong <= elongation <= max_elong:
            cv2.drawContours(output, [c], -1, 255, -1)
    
    return output

# ------------------------------------------------------------
# 7. KONVEXE HÜLLE
# ------------------------------------------------------------
def apply_convex_hull(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(binary_img)
    
    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(output, [hull], -1, 255, -1)
    
    return output

# ------------------------------------------------------------
# 8. KREISERKENNUNG
# ------------------------------------------------------------
def detect_circles(binary_img, min_r=40, max_r=80):
    blurred = cv2.medianBlur(binary_img, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=15,
        minRadius=min_r,
        maxRadius=max_r
    )
    return circles

# ------------------------------------------------------------
# 7. TRACKBAR CALLBACK
# ------------------------------------------------------------
def nothing(x):
    pass


#############################################################


# ------------------------------------------------------------
# 8. HAUPTPROGRAMM
# ------------------------------------------------------------
cap = open_camera_full_reset(0)

cv2.namedWindow("Live Adjustment", cv2.WINDOW_NORMAL)

# Trackbars erstellen
cv2.createTrackbar("Brightness", "Live Adjustment", 3, 255, nothing)
cv2.createTrackbar("Contrast", "Live Adjustment", 1, 89, nothing)
cv2.createTrackbar("Gamma", "Live Adjustment", int(0.3*100), int(10*100), nothing)
cv2.createTrackbar("Threshold", "Live Adjustment", 135, 255, nothing)
cv2.createTrackbar("Morph Size", "Live Adjustment", 21, 100, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Trackbar-Werte lesen
    brightness = cv2.getTrackbarPos("Brightness", "Live Adjustment")
    contrast = cv2.getTrackbarPos("Contrast", "Live Adjustment")
    gamma = cv2.getTrackbarPos("Gamma", "Live Adjustment") / 100.0
    thresh_value = cv2.getTrackbarPos("Threshold", "Live Adjustment")
    #morph_size = cv2.getTrackbarPos("Morph Size", "Live Adjustment")
    #morph_size = max(1, morph_size)  # Kernel darf nicht 0 sein

    # --------------------- Anpassung --------------------------
    adjusted = adjust_brightness_contrast_gamma(frame, brightness, contrast, gamma)
    green_channel = extract_green_channel(adjusted)
    binary = apply_threshold(green_channel, thresh_value)
    area_filtered = filter_by_area(binary, min_area=7000, max_area=18000)
    elongation_filtered = filter_by_elongation(area_filtered, min_elong=0, max_elong=3)
    final_mask = apply_convex_hull(elongation_filtered)
    
    # --------------------- Kreiserkennung ---------------------
    display = frame.copy()
    circles = detect_circles(final_mask)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(display, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(display, (c[0], c[1]), 2, (0, 0, 255), 3)

    # --------------------- Anzeigen ---------------------------
    cv2.imshow("Live Adjustment", green_channel)
    cv2.imshow("Binary", binary)
    cv2.imshow("Area-Filter", area_filtered)
    cv2.imshow("Elongation-Factor", elongation_filtered)
    cv2.imshow("Convexe Hülle", final_mask)
    #cv2.imshow("Detected Circles", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
