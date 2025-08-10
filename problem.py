import cv2
import numpy as np
import mediapipe as mp
import datetime
from collections import deque

# Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

# Mediapipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Canvas & Variables
canvas = None
prev_x, prev_y = 0, 0
draw_color = (255, 0, 0)
brush_thickness = 7
eraser_mode = False
drawing_history = deque(maxlen=50)

# Button Layout (top row and second row)
button_positions = {
    # Color Buttons
    "blue": (10, 10, 60, 60),
    "red": (70, 10, 120, 60),
    "green": (130, 10, 180, 60),
    "erasor": (190, 10, 240, 60),  # was 'black'
    "white": (250, 10, 300, 60),
    "orange": (310, 10, 370, 60),
    "yellow": (380, 10, 440, 60),
    "purple": (450, 10, 510, 60),
    "brown": (520, 10, 580, 60),

    # Tools (Second Row)
    "white": (10, 70, 80, 120),  # was 'eraser'
    "thin": (90, 70, 150, 120),
    "medium": (160, 70, 240, 120),
    "thick": (250, 70, 330, 120),
    "xthick": (340, 70, 430, 120),
    "clear": (440, 70, 510, 120),
    "save": (520, 70, 580, 120),
    "undo": (590, 70, 650, 120)
}

# Draw buttons
def draw_buttons(img):
    for key, (x1, y1, x2, y2) in button_positions.items():
        color = (255, 255, 255)
        label_color = (0, 0, 0)

        if key in ["blue", "red", "green", "erasor", "white", "orange", "yellow", "purple", "brown"]:
            color_dict = {
                "blue": (139, 0, 0),
                "red": (0, 0, 139),
                "green": (0, 100, 0),
                "erasor": (0, 0, 0),  # black color
                "white": (255, 255, 255),
                "orange": (0, 100, 200),
                "yellow": (0, 139, 139),
                "purple": (75, 0, 130),
                "brown": (42, 42, 100),
            }

            color = color_dict[key]
            label_color = (255, 255, 255) if color != (255, 255, 255) else (0, 0, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, key.capitalize(), (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

def check_button_click(x, y):
    global draw_color, eraser_mode, brush_thickness, canvas

    for key, (x1, y1, x2, y2) in button_positions.items():
        if x1 < x < x2 and y1 < y < y2:
            if key in ["blue", "red", "green", "erasor", "white", "orange", "yellow", "purple", "brown"]:
                color_dict = {
                    "blue": (255, 0, 0),
                    "red": (0, 0, 255),
                    "green": (0, 255, 0),
                    "erasor": (0, 0, 0),  # black color
                    "white": (255, 255, 255),
                    "orange": (0, 165, 255),
                    "yellow": (0, 255, 255),
                    "purple": (128, 0, 128),
                    "brown": (42, 42, 165),
                }
                draw_color = color_dict[key]
                eraser_mode = False
            elif key == "white":  # was 'eraser'
                eraser_mode = True
            elif key == "thin":
                brush_thickness = 5
            elif key == "medium":
                brush_thickness = 10
            elif key == "thick":
                brush_thickness = 20
            elif key == "xthick":
                brush_thickness = 30
            elif key == "clear":
                canvas = np.zeros_like(canvas)
                drawing_history.clear()
            elif key == "save":
                filename = "drawing_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
                cv2.imwrite(filename, canvas)
                print(f"✅ Saved as {filename}")
            elif key == "undo":
                if drawing_history:
                    canvas[:] = drawing_history.pop()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_buttons(frame)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) >= 8:
                x1, y1 = lm_list[8]

                # UI click
                check_button_click(x1, y1)

                # Drawing
                fingers_up = lm_list[8][1] < lm_list[6][1]
                if fingers_up and y1 > 130:  # Avoid drawing in button area
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x1, y1

                    color = (255, 255, 255) if eraser_mode else draw_color
                    thickness = 30 if eraser_mode else brush_thickness

                    drawing_history.append(canvas.copy())
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), color, thickness)
                    prev_x, prev_y = x1, y1
                else:
                    prev_x, prev_y = 0, 0

    # Overlay canvas
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Brush Preview
    preview_color = (255, 255, 255) if eraser_mode else draw_color
    cv2.circle(frame, (w - 40, h - 40), brush_thickness, preview_color, -1)
    cv2.putText(frame, "Preview", (w - 110, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, preview_color, 2)

    cv2.imshow("🎨 Virtual Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break  # ESC

cap.release()
cv2.destroyAllWindows()