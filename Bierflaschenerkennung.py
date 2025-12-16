# ------------------------------------------------------------
# 1. Bierflaschenerkennung
# ------------------------------------------------------------

#-----
import cv2
import numpy as np
#-----

# ------------------------------------------------------------
# 1. HELLIGKEIT / KONTRAST / GAMMA
# ------------------------------------------------------------
def adjust_brightness_contrast_gamma(img, brightness=0, contrast=60, gamma=5):
    # Contrast: α = 1 + contrast/100, Brightness: β = brightness
    alpha = 1 + (contrast / 100.0)
    beta = brightness

    # Lineare Helligkeit/Kontrast-Anpassung
    img_lin = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Gamma-Korrektur
    gamma_corrected = np.power(img_lin / 255.0, 1.0 / gamma)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

    return gamma_corrected

# ------------------------------------------------------------
# 2. COLOR PLANE EXTRACTION
#    (Beispiel: Grünkanal extrahieren → kann bei IBV üblich sein)
# ------------------------------------------------------------
def extract_color_plane(img, channel='G'):
    if channel == 'R':
        return img[:, :, 2]
    if channel == 'G':
        return img[:, :, 1]
    if channel == 'B':
        return img[:, :, 0]
    raise ValueError("channel muss 'R', 'G' oder 'B' sein.")

# ------------------------------------------------------------
# 3. MANUELLER THRESHOLD (LOOK FOR BRIGHT OBJECTS)
#    Untergrenze = 190
# ------------------------------------------------------------
def apply_threshold(gray_img, thresh_value=190):
    _, binary = cv2.threshold(gray_img, thresh_value, 255, cv2.THRESH_BINARY)
    return binary

# ------------------------------------------------------------
# 4. MORPHOLOGY – CLOSING (SIZE = 21)
# ------------------------------------------------------------
def apply_morphology(binary_img, size=21):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    return closed

# ------------------------------------------------------------
# 5. PARTICLE FILTERS
#    5.1 AREA (0–100)
#    5.2 ELONGATION (0–5), EXCLUDE INTERVAL = TRUE
# ------------------------------------------------------------
def particle_filters(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(binary_img)

    for c in contours:
        area = cv2.contourArea(c)

        # Filter 1: Area 0–100
        if not (0 <= area <= 100):
            continue

        # Elongation: Verhältnis Hauptachse / Nebenachse
        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            (center, axes, angle) = ellipse
            major = max(axes)
            minor = min(axes)
            elongation = major / minor if minor > 0 else 999
        else:
            elongation = 999

        # Filter 2: Elongation (Ausschlussintervall 0–5)
        # Bedeutet: OBJEKTE innerhalb 0–5 WERDEN EXKLUDIERT
        if 0 <= elongation <= 5:
            continue

        # Objekt akzeptiert → zeichnen
        cv2.drawContours(output, [c], -1, 255, -1)

    return output

# ------------------------------------------------------------
# 6. KREISERKENNUNG (HoughCircles)
#    Radiusbereich 40–80
# ------------------------------------------------------------
def detect_circles(binary_img, min_r=40, max_r=80):
    # Für Hough müssen wir ein Graubild haben
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
# HAUPTPROGRAMM
# ------------------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --------------------- Schritt 1 --------------------------
    adjusted = adjust_brightness_contrast_gamma(frame)

    # --------------------- Schritt 2 --------------------------
    extracted = extract_color_plane(adjusted, channel='G')

    # --------------------- Schritt 3 --------------------------
    binary = apply_threshold(extracted, thresh_value=190)

    # --------------------- Schritt 4 --------------------------
    closed = apply_morphology(binary, size=21)

    # --------------------- Schritt 5 --------------------------
    particles = particle_filters(closed)

    # --------------------- Schritt 6 --------------------------
    circles = detect_circles(particles)

    display = frame.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(display, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(display, (c[0], c[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Original", frame)
    cv2.imshow("Binary", binary)
    cv2.imshow("Particles", particles)
    cv2.imshow("Detected Circles", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()