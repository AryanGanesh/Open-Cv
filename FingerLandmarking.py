import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_styles = mp.solutions.drawing_styles

# Create custom DrawingSpec for all landmarks and all connections
landmark_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=1) 
connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2) 


'''
#if using custom colors for custom parts
# Get default styles
landmark_annotations = mp_styles.get_default_hand_landmarks_style()
connection_annotations = mp_styles.get_default_hand_connections_style()

# Customize the wrist landmark (Point 0)
custom_color_wrist = (0, 255, 0)  # (B, G, R)
landmark_annotations[mp_hands.HandLandmark.] = mp_drawing.DrawingSpec(
    color=custom_color_wrist, thickness=3, circle_radius=3
)

# Customize the connection between PINKY_DIP and PINKY_TIP (Points 19 and 20)
custom_color_pinky = (0, 255, 0)  # (B, G, R)
connection_annotations[
    (mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP)
] = mp_drawing.DrawingSpec(
    color=custom_color_pinky, thickness=10
)
'''
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.8
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip image for selfie view and convert to RGB
        image = cv2.flip(image, 1)                   
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        '''
        #if using custom colors for custom parts 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,  
                    landmark_drawing_spec=landmark_annotations,
                    connection_drawing_spec=connection_annotations
                )
        '''
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec
                )
                
        cv2.imshow('MediaPipe Hands - Custom Styles', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
