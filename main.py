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



class FaceRecApp(App):

    def build(self):
        layout = BoxLayout(orientation = "vertical")
        self.image = Image()
        layout.add_widget(self.image)
        layout.add_widget(Button(
            text = "Choose ratio",
            pos_hint = {"center_x": .5, "center_y": .5},
            size_hint = (None, None), 
            size = ("200dp", "100dp")
        ))
        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        return layout


    def load_video(self, *args):
        """ret, frame = self.cap.read()
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture"""

        def eye_aspect_ratio(eye):
            A = distance.euclidean(eye[1], eye[5])
            B = distance.euclidean(eye[2], eye[4])
            C = distance.euclidean(eye[0], eye[3])
            ear = (A + B) / (2.0 * C)
            return ear

        thresh = 0.25
        frame_check = 20
        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


        cap=cv2.VideoCapture(0)
        flag=0
        thresh = 0
        while True:   
            ret, frame=cap.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)
            for subject in subjects:
                #print (thresh)
                if thresh == 0:
                    cv2.putText(frame, "thresh is {} Press s to establish your eye threshhold".format(thresh), (30,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    thresh = ear
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)#converting to NumPy Array
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear < thresh:
                    flag += 1
                    #print (flag)
                    if flag >= frame_check:
                        cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "****************ALERT!****************", (10,325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    flag = 0
            cv2.imshow("Frame", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                cap.release()
                break


    





if __name__ == '__main__':
    FaceRecApp().run()