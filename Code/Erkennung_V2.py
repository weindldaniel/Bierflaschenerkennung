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
def detect_circles(binary_img, min_r, max_r):
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

# Trackbars erstellen um die Werte vom VisionAssistant nachzubilden
cv2.createTrackbar("Brightness", "Live Adjustment", 3, 255, nothing)
cv2.createTrackbar("Contrast", "Live Adjustment", 1, 89, nothing)
cv2.createTrackbar("Gamma", "Live Adjustment", int(0.3*100), int(10*100), nothing)
cv2.createTrackbar("Threshold", "Live Adjustment", 243, 255, nothing)
#---
cv2.createTrackbar("MinArea", "Live Adjustment", 7000, 50000, nothing)
cv2.createTrackbar("MaxArea", "Live Adjustment", 32000, 50000, nothing)
#--
cv2.createTrackbar("minR", "Live Adjustment", 40, 200, nothing)
cv2.createTrackbar("maxR", "Live Adjustment", 105, 200, nothing)
#---


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # live Trackbar-Werte einlesen
    brightness = cv2.getTrackbarPos("Brightness", "Live Adjustment")
    contrast = cv2.getTrackbarPos("Contrast", "Live Adjustment")
    gamma = cv2.getTrackbarPos("Gamma", "Live Adjustment") / 100.0
    thresh_value = cv2.getTrackbarPos("Threshold", "Live Adjustment")
    #---
    min_area = cv2.getTrackbarPos("MinArea", "Live Adjustment")
    max_area = cv2.getTrackbarPos("MaxArea", "Live Adjustment")
    #---
    minR = cv2.getTrackbarPos("minR", "Live Adjustment")
    maxR = cv2.getTrackbarPos("maxR", "Live Adjustment")
    
    # --------------------- Filter Anwenden --------------------------
    adjusted = adjust_brightness_contrast_gamma(frame, brightness, contrast, gamma)
    green_channel = extract_green_channel(adjusted)
    binary = apply_threshold(green_channel, thresh_value)
    #---
    area_filtered = filter_by_area(binary, min_area, max_area)
    #elongation_filtered = filter_by_elongation(binary, min_elong=0, max_elong=3) # ist überflüssig weil Area-Filtern besser funktioniert als im VisonAssistant
    final_mask = apply_convex_hull(area_filtered)
        
    # --------------------- Kreiserkennung ---------------------
    display = frame.copy()
    
    # Sicherstellen, dass Farbbild vorliegt
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    
    circles = detect_circles(final_mask, minR, maxR)
    
    # Anzahl initialisieren
    bottle_count = 0
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        bottle_count = len(circles[0])
    
        for x, y, r in circles[0]:
            # Roter Kreis
            cv2.circle(display, (x, y), r, (0, 0, 255), 2)
    
            # Rotes X im Mittelpunkt
            size = int(r * 0.15)
            cv2.line(display, (x - size, y - size),
                              (x + size, y + size), (0, 0, 255), 2)
            cv2.line(display, (x - size, y + size),
                              (x + size, y - size), (0, 0, 255), 2)
    
    # --------------------- Text Overlay ---------------------
    text = f"Bierflaschen: {bottle_count}"
    
    cv2.putText(
        display,
        text,
        (30, 60),                       # Position
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,                            # Schriftgröße
        (0, 0, 255),                    # Rot
        3,                              # Linienstärke
        cv2.LINE_AA
    )


    # --------------------- Anzeigen ---------------------------
    cv2.imshow("Live Adjustment", final_mask)
    cv2.imshow("Kreisdetektion", display)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()