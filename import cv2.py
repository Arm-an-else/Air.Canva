import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize color points and indices
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Colors and thicknesses
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0
thicknessIndex = 0
thicknesses = [2, 5, 10, 15]

# Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255

# Tkinter App
class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Paint App")

        # Create a canvas for the paint window
        self.canvas = tk.Canvas(root, width=636, height=471, bg="white")
        self.canvas.pack()

        # Create a button to clear the canvas
        self.clear_button = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack()

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.update()

    def clear_canvas(self):
        global bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index, paintWindow
        bpoints = [deque(maxlen=512)]
        gpoints = [deque(maxlen=512)]
        rpoints = [deque(maxlen=512)]
        ypoints = [deque(maxlen=512)]

        blue_index = 0
        green_index = 0
        red_index = 0
        yellow_index = 0

        paintWindow = np.zeros((471, 636, 3)) + 255
        self.canvas.delete("all")

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks
            result = hands.process(framergb)
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * 640)
                        lmy = int(lm.y * 480)
                        landmarks.append([lmx, lmy])

                    # Draw landmarks on the frame
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                fore_finger = (landmarks[8][0], landmarks[8][1])
                center = fore_finger
                thumb = (landmarks[4][0], landmarks[4][1])

                if thumb[1] - center[1] < 30:
                    bpoints.append(deque(maxlen=512))
                    blue_index += 1
                    gpoints.append(deque(maxlen=512))
                    green_index += 1
                    rpoints.append(deque(maxlen=512))
                    red_index += 1
                    ypoints.append(deque(maxlen=512))
                    yellow_index += 1

                elif center[1] <= 40:
                    if 40 <= center[0] <= 60:  # Clear Button
                        self.clear_canvas()
                    elif 60 <= center[0] <= 80:
                        colorIndex = 0  # Blue
                    elif 80 <= center[0] <= 100:
                        colorIndex = 1  # Green
                    elif 100 <= center[0] <= 120:
                        colorIndex = 2  # Red
                    elif 120 <= center[0] <= 140:
                        colorIndex = 3  # Yellow
                elif 450 <= center[1] <= 480:
                    if 0 <= center[0] <= 50:
                        thicknessIndex = 0
                    elif 50 <= center[0] <= 100:
                        thicknessIndex = 1
                    elif 100 <= center[0] <= 150:
                        thicknessIndex = 2
                    elif 150 <= center[0] <= 200:
                        thicknessIndex = 3
                else:
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(center)

            # Draw lines on the canvas
            points = [bpoints, gpoints, rpoints, ypoints]
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], thicknesses[thicknessIndex])
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], thicknesses[thicknessIndex])

            # Convert the paint window to an image and display it on the Tkinter canvas
            img = Image.fromarray(cv2.cvtColor(paintWindow, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        self.root.after(10, self.update)

    def __del__(self):
        self.cap.release()

# Run the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()