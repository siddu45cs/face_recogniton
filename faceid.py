# import kivy dependencies 
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout 

# import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

#import other kivy stuffs
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# import other dependences
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

#build app ad layout
class CamApp(App):

    def build(self):
        # main layout components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify",on_press = self.verify,size_hint=(1,.1))
        self.verification_label = Label(text="verification uninitiated", size_hint=(1,.1))

        # add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        # setup vdo capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
       
        return layout
    
    # run continuously to get webcam feed
    def update(self, *args):
        
        # read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # flip horizantal and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
        

    def preprocess(self,file_path):
        #read img from file path
        byte_img = tf.io.read_file(file_path)
        #load in the img
        img = tf.io.decode_jpeg(byte_img)
        #pre[rocessing steps resizimg the img 
        img = tf.image.resize(img, (100,100))
        # scale img to be between 0 and 1
        img = img / 255.0
        #return img
        return img
    
    #  verification function to verify person
    def verify( self,*args):
        detection_treshold = 0.4
        verification_treshold = 0.4

        # capture input img from webcwm
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

    #built result array
        results=[]
        for image in os.listdir(os.path.join('application_data', 'verification_image')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_image', image))

            #make prediction
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # detection_treshold metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_treshold)

        # verification treshold: prportion of positive prediction / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_image')))
        verified = verification > verification_treshold

        # set verification text
        self.verification_label.text = 'verified' if verified == True else 'unverified'

        # log out details
        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.3))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))

        return results,verified
        
    
    

if __name__ == '__main__':
    CamApp().run()
        
    