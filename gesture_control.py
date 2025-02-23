import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)
prev_click_time = time.time()  # Prevents unintended clicks
prev_double_click_time = time.time()  # Prevents unintended double clicks

middle_was_bent = True  # Track middle finger previous state
ring_was_bent = True  # Track ring finger previous state

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand tracking
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract fingertip positions
            index_tip = hand_landmarks.landmark[8]  # Index finger tip
            index_mcp = hand_landmarks.landmark[5]  # Index finger base
            middle_tip = hand_landmarks.landmark[12]  # Middle finger tip
            middle_mcp = hand_landmarks.landmark[9]  # Middle finger base
            ring_tip = hand_landmarks.landmark[16]  # Ring finger tip
            ring_mcp = hand_landmarks.landmark[13]  # Ring finger base

            # Convert to screen coordinates
            x = int(index_tip.x * screen_width)
            y = int(index_tip.y * screen_height)

            # Check if index finger is extended
            index_straight = index_tip.y < index_mcp.y  # Tip is above base

            # Check if middle finger is extended
            middle_straight = middle_tip.y < middle_mcp.y  # Tip is above base

            # Check if ring finger is extended
            ring_straight = ring_tip.y < ring_mcp.y  # Tip is above base

            # Cursor movement: Move if only index finger is straight
            if index_straight and not middle_straight and not ring_straight:
                pyautogui.moveTo(x, y, duration=0.02)  # Smooth movement

            current_time = time.time()

            # Click condition: Middle finger extends AFTER being bent
            if index_straight and middle_straight and middle_was_bent and (current_time - prev_click_time) > 0.5:
                pyautogui.click()
                print("Click!")
                prev_click_time = current_time  # Update last click time

            # Double-click condition: Ring finger also extends AFTER being bent
            if index_straight and middle_straight and ring_straight and ring_was_bent and (current_time - prev_double_click_time) > 0.7:
                pyautogui.doubleClick()
                print("Double Click!")
                prev_double_click_time = current_time  # Update last double-click time

            # Update finger states for next iteration
            middle_was_bent = not middle_straight  # True if it was bent before
            ring_was_bent = not ring_straight  # True if it was bent before

    # Display webcam feed
    cv2.imshow("Hand Tracking - Cursor & Click", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
