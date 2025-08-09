from kivymd.icon_definitions import md_icons
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.image import Image as KivyImage
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.modalview import ModalView
import os
import cv2
import numpy as np
import threading
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from utils.tflite_predictor import WeedDetector
from kivymd.uix.fitimage import FitImage
from kivy.utils import platform
import sys
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
from kivy.animation import Animation
from kivymd.uix.button import MDButton, MDButtonText, MDButtonIcon

# Add splash screen handling
if getattr(sys, 'frozen', False):
    import pyi_splash
    pyi_splash.update_text("Loading weed detector...")
    pyi_splash.update_text("Initializing camera...")
    pyi_splash.update_text("Loading AI model...")
    pyi_splash.update_text("Starting application...")

KV = '''
#:import images_path kivymd.images_path

MDScreen:
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(20)
        spacing: dp(10)

        MDCard:
            size_hint: 1, 0.6
            pos_hint: {'center_x': 0.5, 'center_y': 0.5}
            style: 'elevated'
            elevation_level: 2  

            MDBoxLayout:
                orientation: 'vertical'
                FitImage:
                    id: video_feed
                    source: ''
                    radius: "36dp", "36dp", "36dp", "36dp"
                    size_hint: 1, 1
                    pos_hint: {'center_x': 0.5, 'center_y': 0.5}

        MDBoxLayout:
            orientation: 'vertical'
            adaptive_height: True
            padding: 10
            spacing: 5

            MDIconButton:
                id: start_camera_button
                pos_hint: {'center_x': 0.5}
                icon: 'camera'
                style: 'standard'
                on_release: 
                    app.start_camera_detection()

            MDButton:
                pos_hint: {'center_x': 0.5}
                on_release: app.pick_video_file()

                MDButtonText:
                    text: 'Detect from video'

                MDButtonIcon:
                    icon: 'folder'

            MDButton:
                pos_hint: {'center_x': 0.5}
                on_release: app.pick_image_file()

                MDButtonText:
                    text: 'Detect from image'

                MDButtonIcon:
                    icon: 'image'

            MDButton:
                pos_hint: {'center_x': 0.5}
                on_release: app.show_metrics()

                MDButtonText:
                    text: 'View Metrics'

                MDButtonIcon:
                    icon: 'chart-bar'
'''

if platform == 'android':
    from android.permissions import request_permissions, Permission
    def request_android_permissions():
        request_permissions([Permission.CAMERA, Permission.READ_EXTERNAL_STORAGE,
                             Permission.WRITE_EXTERNAL_STORAGE])
else:
    def request_android_permissions():
        pass

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class FavourBroadLeafDetectorApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera_active = False
        self.latest_probabilities = None
        self.metrics_popup = None

    def build(self):
        if getattr(sys, 'frozen', False):
            pyi_splash.update_text("Kwara State University CSC499\nWeed Detection System")
            Clock.schedule_once(lambda dt: pyi_splash.close(), 2)
            
        self.icon = resource_path('assets/icon.ico')
        request_android_permissions()
        self.title = "Favors CNN Weed Detector"
        model_path = resource_path('assets/weed_detector.tflite')
        self.detector = WeedDetector(model_path)
        self.interpreter = self.detector.interpreter
        self.input_details = self.detector.input_details
        self.output_details = self.detector.output_details
        self.labels = self.detector.classes
        self.icon = resource_path('assets/icon.ico')
        return Builder.load_string(KV)

    def start_camera_detection(self):
        if not self.check_opencv_config():
            self.show_error_popup("OpenCV Error", 
                "OpenCV configuration files missing. Please check installation.")
            return
        
        try:
            if not self.camera_active:
                self.capture = cv2.VideoCapture(0)
                if not self.capture.isOpened():
                    self.show_error_popup("Camera Error", 
                        "Could not access the camera. Please check your camera connection.")
                    return
                self.camera_active = True
                Clock.schedule_interval(self.update_camera, 1.0 / 30.0)
                self.root.ids.start_camera_button.icon = 'stop'
            else:
                self.stop_camera_detection()
        except Exception as e:
            self.show_error_popup("Camera Error", f"Error accessing camera: {str(e)}")

    def stop_camera_detection(self):
        if self.camera_active:
            self.camera_active = False
            Clock.unschedule(self.update_camera)
            self.capture.release()
            self.root.ids.video_feed.texture = None
            self.root.ids.start_camera_button.icon = 'camera'

    def update_camera(self, dt):
        if self.capture.isOpened() and self.camera_active:
            ret, frame = self.capture.read()
            if ret:
                img = cv2.resize(frame, (128, 128))
                img = img.astype('float32') / 255.0
                input_data = np.expand_dims(img, axis=0)

                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

                predicted_index = np.argmax(output_data[0])
                predicted_label = self.labels[predicted_index]
                self.latest_probabilities = {label: float(output_data[0][i]) for i, label in enumerate(self.labels)}

                frame_height, frame_width, _ = frame.shape
                text_x = int(frame_width / 2)
                text_y = int(frame_height / 2)
                cv2.putText(frame, predicted_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                buf = cv2.flip(frame, 0).tobytes()
                image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.root.ids.video_feed.texture = image_texture

    def pick_video_file(self):
        filechooser = FileChooserIconView(path=os.getcwd(), filters=["*.mp4"])
        popup = Popup(title="Select a Video", content=filechooser, size_hint=(0.9, 0.9))
        filechooser.bind(on_submit=lambda fc, selection, touch: self.load_selected_video(selection, popup))
        popup.open()

    def pick_image_file(self):
        filechooser = FileChooserIconView(path=os.getcwd(), filters=["*.jpg", "*.jpeg", "*.png", "*.tif"])
        popup = Popup(title="Select an Image", content=filechooser, size_hint=(0.9, 0.9))
        filechooser.bind(on_submit=lambda fc, selection, touch: self.load_selected_image(selection, popup))
        popup.open()

    def load_selected_image(self, selection, popup):
        if selection:
            selected_path = selection[0]
            print(f"🖼️ Selected image: {selected_path}")
            self.detect_from_image(selected_path)
        popup.dismiss()

    def detect_from_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.show_error_popup("Image Error", "Failed to load image.")
                return

            buf = cv2.flip(img, 0).tobytes()
            image_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.root.ids.video_feed.texture = image_texture

            img_resized = cv2.resize(img, (128, 128))
            img_normalized = img_resized.astype('float32') / 255.0
            input_data = np.expand_dims(img_normalized, axis=0)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            predicted_index = np.argmax(output_data[0])
            predicted_label = self.labels[predicted_index]
            self.latest_probabilities = {label: float(output_data[0][i]) for i, label in enumerate(self.labels)}

            frame_height, frame_width, _ = img.shape
            text_x = int(frame_width / 2)
            text_y = int(frame_height / 2)
            cv2.putText(img, predicted_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            buf = cv2.flip(img, 0).tobytes()
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.root.ids.video_feed.texture = image_texture

        except Exception as e:
            self.show_error_popup("Image Error", f"Error processing image: {str(e)}")

    def show_metrics(self):
        if not self.latest_probabilities:
            self.show_error_popup("No Metrics", "No prediction data available. Please run detection first.")
            return

        self.metrics_popup = ModalView(size_hint=(1, 0.45), pos_hint={'top': 0.45}, background_color=(0, 0, 0, 0))
        
        layout = MDBoxLayout(orientation='vertical', padding=5, spacing=5)
        
        plt.figure(figsize=(5, 2.5))  # Reduced figure size
        plt.bar(self.latest_probabilities.keys(), self.latest_probabilities.values(), 
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.xlabel('Classes')
        plt.ylabel('Probabilities')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')  # Align x-axis labels to prevent cutoff
        plt.tight_layout(pad=0.5)  # Adjust layout to fit labels
        
        chart_widget = FigureCanvasKivyAgg(plt.gcf())
        layout.add_widget(chart_widget)
        
        close_button = MDButton(
            pos_hint={'center_x': 0.5},
            size_hint=(0.5, None),
            height=30
        )
        close_button.add_widget(MDButtonText(text='Close Metrics'))
        close_button.add_widget(MDButtonIcon(icon='close'))
        layout.add_widget(close_button)
        
        self.metrics_popup.add_widget(layout)
        
        self.metrics_popup.pos_hint = {'top': 0}
        self.metrics_popup.open()
        Animation(pos_hint={'top': 0.45}, duration=0.3).start(self.metrics_popup)

    def close_metrics(self, *args):
        if self.metrics_popup:
            Animation(pos_hint={'top': 0}, duration=0.3).start(self.metrics_popup)
            Clock.schedule_once(lambda dt: self.metrics_popup.dismiss(), 0.3)
            plt.close('all')

    def detect_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('annotated_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        def process_frame(dt):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                out.release()
                print("Detection completed. Annotated video saved as 'annotated_output.mp4'")
                Clock.unschedule(process_frame)
                return

            img = cv2.resize(frame, (128, 128))
            img = img.astype('float32') / 255.0
            input_data = np.expand_dims(img, axis=0)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            predicted_index = np.argmax(output_data[0])
            predicted_label = self.labels[predicted_index]
            confidence = output_data[0][predicted_index] * 100
            self.latest_probabilities = {label: float(output_data[0][i]) for i, label in enumerate(self.labels)}

            display_text = f"{predicted_label}: {confidence:.2f}%"
            font_scale = 3.0
            thickness = 4
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
            frame_height, frame_width, _ = frame.shape
            text_x = int((frame_width - text_width) / 2)
            text_y = int((frame_height + text_height) / 2)
            cv2.putText(frame, display_text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
            out.write(frame)

            buf = cv2.flip(frame, 0).tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.root.ids.video_feed.texture = image_texture

        Clock.schedule_interval(process_frame, 1.0 / 30.0)

    def load_selected_video(self, selection, popup):
        if selection:
            selected_path = selection[0]
            print(f"🎞️ Selected video: {selected_path}")
            self.detect_from_video(selected_path)
        popup.dismiss()

    def show_logs(self):
        print("📜 Logs feature will be added soon...")

    def show_error_popup(self, title, message):
        popup = Popup(
            title=title,
            content=MDLabel(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()

    def on_stop(self):
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
        if hasattr(self, 'interpreter'):
            del self.interpreter
        if self.metrics_popup:
            self.metrics_popup.dismiss()
        plt.close('all')
        cv2.destroyAllWindows()

    def check_opencv_config(self):
        try:
            temp_cap = cv2.VideoCapture(0)
            if temp_cap is None or not temp_cap.isOpened():
                return False
            temp_cap.release()
            return True
        except Exception:
            return False

if __name__ == '__main__':
    FavourBroadLeafDetectorApp().run()