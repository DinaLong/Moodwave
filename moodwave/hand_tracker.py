import cv2
import mediapipe as mp

class Tracker():
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(static_image_mode=False,
                                              max_num_hands=1, 
                                              min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
        self.tracking_id = [8]
           
    def hand_landmark(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        return img
    
    def tracking(self, img):
        tracking_points = []
        x1 = -1
        y1 = -1
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(hand_landmarks.landmark):
                if id in self.tracking_id:
                    h, w, c = img.shape
                    x, y = int(lm.x*w), int(lm.y*h)
                    tracking_points.append((x, y))
                    cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)  
            x1, y1 = tracking_points[0]      
        
        return img, x1, y1
    
