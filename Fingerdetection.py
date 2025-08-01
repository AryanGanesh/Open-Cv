import cv2
import mediapipe as mp

# Calling drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_styles = mp.solutions.drawing_styles

# Specifying connection and landmarks sizes and colours
landmark_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=1) 
connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2) 

# Initialize Mediaknuckle Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.95
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    
    # Convert to RGB and process
    frame=cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec
                )
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get image dimensions
            h, w, _ = frame.shape
            
            # Extract landmark coordinates
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            # Calculate bounding box
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)
            
            cv2.rectangle(frame, (x_min - 40, y_min - 40), (x_max + 40, y_max + 40), (0, 255, 0), 2)
            
            # Detect raised fingers
            finger_tips = [4, 8, 12, 16, 20]
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            raised_fingers = []

            # Thumb detection (compare x-coordinates)
            if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_tips[0] - 1].x:
                raised_fingers.append(finger_names[0])
            
            # Other fingers detection (compare y-coordinates)
            for i in range(1, 5):
                tip = hand_landmarks.landmark[finger_tips[i]]
                knuckle= hand_landmarks.landmark[finger_tips[i] - 2]
                if tip.y < knuckle.y:  # Tip above knuckle
                     raised_fingers.append(finger_names[i])
            
            # Prepare finger status text
            status_text = ', '.join(raised_fingers) if raised_fingers else "No fingers up"
            
            # Display text above bounding box
            text_position = (x_min, max(y_min - 10, 20))
            cv2.putText(frame, status_text, text_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Finger Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
