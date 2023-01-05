from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import dlib 
import imutils
from scipy.spatial import distance
from imutils import face_utils
from kivy.core.audio import SoundLoader
import numpy as np





class FaceRecApp(App):

    def build(self):

        self.sound = SoundLoader.load('bleep.mp3')
        self.running = True
        self.thresh = 0
        self.ear = 0
        self.flag = 0
        self.frame_check = 40
        self.allowed_out_time = 70
        self.allowed_stop_time = 70
        self.in_frame = False
        self.no_face = False
        self.time_out_frame = 0
        self.stop_time = 0
        self.closed_eyes = False
        self.min_amplitude = 2
        self.not_move = False
        self.stoned = False
        self.prev_nose = np.zeros((9,2))


        layout = BoxLayout(orientation = "vertical")
        self.image = Image()
        layout.add_widget(self.image)

        self.buttonSetRatio = Button(
            text = "Choose ratio",
            pos_hint = {"center_x": .5, "center_y": .5},
            size_hint = (None, None), 
            size = ("200dp", "100dp")
        )
        self.buttonQuit = Button(
            text = "Quit",
            pos_hint = {"center_x": .5, "center_y": .5},
            size_hint = (None, None), 
            size = ("200dp", "100dp")
        )

        self.buttonSetRatio.bind(on_press = self.adjust)
        self.buttonQuit.bind(on_press = self.quit)

        #PRESET OF NEEDED VARIABLES AND MODELS
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (self.nStart, self.nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

        layout.add_widget(self.buttonSetRatio)
        layout.add_widget(self.buttonQuit)

        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        return layout

    def adjust(self, *args):
        self.thresh = self.ear

    
    def quit(self, *args):
        cv2.destroyAllWindows()
        self.cap.release()
        self.running = False

    def playsound(self, *args):
        self.sound.play()

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear  

    

    def load_video(self, *args):

        ret, frame = self.cap.read()
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 0)
        self.in_frame = False
        for subject in subjects:
            if self.thresh == 0:
                cv2.putText(frame, "thresh is {} Press s to establish your eye threshhold".format(self.thresh), (30,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            shape = self.predict(gray, subject)
            shape = face_utils.shape_to_np(shape)#converting to NumPy Array

            self.in_frame = True

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth = shape[self.mStart:self.mEnd]
            nose = shape[self.nStart:self.nEnd]
            #print(nose)

            diff = np.subtract(self.prev_nose, nose)
            #print('init diff is ', diff)
            diff = np.array(list(map(abs, diff)))
            #diff = np.fromiter(seq, dtype = np.int)
            #print('abs diff is ', diff)
            average = np.sum(diff)/diff.size
            print('average is', average)
            self.prev_nose = nose
            if average <= self.min_amplitude:
                self.stop_time += 1
                if self.stop_time >= self.allowed_stop_time:
                    self.stoned = True
            else:
                self.stop_time = 0
                self.stoned = False


            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            self.ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            noseHull = cv2.convexHull(nose)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

            
            if self.ear < self.thresh:
                self.flag += 1
                if self.flag >= self.frame_check:
                    self.closed_eyes = True
            else:
                self.flag = 0
                self.closed_eyes = False

        if not self.in_frame:
            self.time_out_frame += 1
            print('out of frame ', self.time_out_frame)
            if self.time_out_frame >= self.allowed_out_time:
                self.no_face = True
        else:
            self.time_out_frame = 0
            self.no_face = False

        if self.no_face or self.closed_eyes or self.stoned:
            cv2.putText(frame, "****************ALERT!****************", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "****************ALERT!****************", (10,325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.sound.state == 'stop':
                self.sound.play()
        else:
            if self.sound.state == 'play':
                self.sound.stop()

        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

        
            


if __name__ == '__main__':
    FaceRecApp().run()