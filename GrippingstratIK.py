import cv2
import numpy as np

# ---- Finger Segment Lengths (cm, adjust to your hand) ----
L1, L2 = 5.0, 4.0  # Proximal and distal phalanges

# ---- Finger Base Positions (pixel coordinates in camera frame) ----
FINGER_BASES = [
    (180, 470),  # Thumb
    (240, 470),  # Index
    (320, 470),  # Middle
    (400, 470),  # Ring
    (460, 470),  # Little
]

PX_TO_CM = 0.05  # Pixel to cm conversion (calibrate for your setup)

def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)
    shape = "unidentified"
    if vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif vertices == 5:
        shape = "pentagon"
    elif vertices == 6:
        shape = "hexagon"
    elif vertices > 6:
        if len(contour) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            if MA and ma:
                eccentricity = np.sqrt(1 - (MA / ma) ** 2) if ma != 0 else 0
                shape = "circle" if 0.0 <= eccentricity < 0.4 else "ellipse"
            else:
                shape = "circle"
        else:
            shape = "circle"
    return shape

def get_evenly_spaced_points(contour, num_points):
    contour = contour.reshape(-1, 2)
    dists = np.zeros(contour.shape[0], dtype=np.float32)
    for i in range(1, len(contour)):
        dists[i] = dists[i-1] + np.linalg.norm(contour[i] - contour[i-1])
    total_length = dists[-1] + np.linalg.norm(contour[0] - contour[-1])
    distances = np.linspace(0, total_length, num_points+1)[:-1]
    points = []
    for d in distances:
        idx = np.searchsorted(dists, d) % len(contour)
        prev_idx = (idx - 1) % len(contour)
        segment_length = dists[idx] - dists[prev_idx]
        if segment_length == 0:
            points.append(tuple(contour[idx]))
            continue
        ratio = (d - dists[prev_idx]) / segment_length
        pt = (1 - ratio) * contour[prev_idx] + ratio * contour[idx]
        points.append(tuple(pt.astype(int)))
    return points

def finger_ik(x, y, L1, L2):
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if np.abs(D) > 1:
        return None  # Target is out of reach
    theta2 = np.arccos(D)
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    return np.degrees(theta1), np.degrees(theta2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            shape = detect_shape(c)
            targets = get_evenly_spaced_points(c, 5)
            cv2.drawContours(frame, [c], -1, (0,255,255), 2)
            for idx, (base, target) in enumerate(zip(FINGER_BASES, targets)):
                cv2.circle(frame, base, 6, (255,0,0), -1)
                cv2.circle(frame, target, 6, (0,0,255), -1)
                cv2.line(frame, base, target, (0,255,0), 2)
                dx = (target[0] - base[0]) * PX_TO_CM
                dy = (base[1] - target[1]) * PX_TO_CM  # y axis inverted in image
                ik_result = finger_ik(dx, dy, L1, L2)
                if ik_result:
                    theta1, theta2 = ik_result
                    cv2.putText(frame, f"{idx+1}:{theta1:.0f},{theta2:.0f}", (base[0]-25, base[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    print(f"Finger {idx+1} | Target: ({dx:.2f}cm, {dy:.2f}cm) | Angles: [{theta1:.1f}, {theta2:.1f}]")
                else:
                    cv2.putText(frame, f"{idx+1}:out", (base[0]-25, base[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    print(f"Finger {idx+1} | Target out of reach")
            # Draw shape label at contour centroid
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        else:
            cv2.putText(frame, "No significant object detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        cv2.putText(frame, "No object detected", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Real-Time IK Grasping (All Fingers)", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
