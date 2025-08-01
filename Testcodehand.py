import cv2
import numpy as np
#import serial  # Only if using serial to communicate with Arduino/MRL

# Ball and camera parameters
REAL_BALL_RADIUS_CM = 3.35  # e.g., tennis ball
FOCAL_LENGTH_PIXELS = 800   # calibrate for your camera!
CAMERA_CX = 540             # image center x (for 1080p)
CAMERA_CY = 360             # image center y (for 720p)

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([29, 86, 6])
    upper_hsv = np.array([64, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center, radius = None, 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            center = (int(x), int(y))
    return center, radius

def estimate_distance(radius_px):
    if radius_px > 0:
        return (FOCAL_LENGTH_PIXELS * REAL_BALL_RADIUS_CM) / radius_px
    else:
        return None

def image_to_camera_coords(center, distance_cm):
    x_img, y_img = center
    x_cam = (x_img - CAMERA_CX) * distance_cm / FOCAL_LENGTH_PIXELS
    y_cam = (y_img - CAMERA_CY) * distance_cm / FOCAL_LENGTH_PIXELS
    z_cam = distance_cm
    return np.array([x_cam, y_cam, z_cam])  # in cm

def plan_finger_positions(radius_px):
    # Map ball size to finger opening (0=closed, 1=fully open)
    max_radius = 50  # adjust for your setup
    opening = min(radius_px / max_radius, 1.0)
    # InMoov fingers typically accept values 0 (open) to 180 (closed)
    finger_angle = int(180 - opening * 120)  # 60 deg open to 180 deg closed
    return finger_angle

def plan_wrist_orientation(target_xyz):
    # For a simple approach, keep wrist flat (90 deg)
    wrist_angle = 90
    return wrist_angle

# If using serial to Arduino/MRL, setup serial port (adjust 'COM4' and baudrate as needed)
# ser = serial.Serial('COM4', 57600, timeout=1)
# def send_arduino_commands(finger_angle, wrist_angle, ser):
#     cmd = f"T{finger_angle}I{finger_angle}M{finger_angle}R{finger_angle}P{finger_angle}W{wrist_angle}\n"
#     ser.write(cmd.encode())

# If using MRL Python/Jython, see next section for direct hand commands.

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    center, radius = detect_ball(frame)
    if center and radius > 10:
        distance_cm = estimate_distance(radius)
        target_xyz = image_to_camera_coords(center, distance_cm)
        finger_angle = plan_finger_positions(radius)
        wrist_angle = plan_wrist_orientation(target_xyz)
        
        # Send commands to InMoov (see next section for MRL Python code)
        # send_arduino_commands(finger_angle, wrist_angle, ser)
        # Or, if using network, send (finger_angle, wrist_angle) to MRL Python script
        
        cv2.circle(frame, center, int(radius), (0, 255, 255), 3)
        cv2.putText(frame, f"{distance_cm:.1f}cm", (center[0], center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("InMoov Ball Grasping", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
