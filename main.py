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
from kivy.utils import platform  # Import platform
import sys

# Add splash screen handling at the start of your file
if getattr(sys, 'frozen', False):
    import pyi_splash
    # Add custom loading messages
    pyi_splash.update_text("Loading weed detector...")
    pyi_splash.update_text("Initializing camera...")
    pyi_splash.update_text("Loading AI model...")
    pyi_splash.update_text("Starting application...")

KV = '''
#:import images_path kivymd.images_path

MDScreen:
    MDBoxLayout:
        orientation: 'vertical'
        # padding: dp(20)
        # spacing: dp(20)

        MDCard:
            size_hint: 1, 1
            pos_hint: {'center_x': 0.5, 'center_y': 0.5}
            style: 'elevated'
            elevation_level: 2  

            MDBoxLayout:
                orientation: 'vertical'
                FitImage:
                    id: video_feed
                    source: ''
                    radius: "36dp", "36dp", "36dp", "36dp"
                    size_hint: 1, 1  # Fill the available space (full width and height)
                    pos_hint: {'center_x': 0.5, 'center_y': 0.5}  # Center the image

                MDBoxLayout:
                    orientation: 'vertical'
                    adaptive_height: True
                    # padding: 10
                    spacing: 5
                    # pos_hint: {'center_x': 0.5}
                
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
                        on_release: app.show_logs()

                        MDButtonText:
                            text: 'view logs'

                        MDButtonIcon:
                            icon: 'file'
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
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class FavourBroadLeafDetectorApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera_active = False
        
    def build(self):
        # Close splash screen if running as frozen app
        if getattr(sys, 'frozen', False):
            pyi_splash.update_text("Kwara State University CSC499\nWeed Detection System")
            # Add a small delay to show the final message
            Clock.schedule_once(lambda dt: pyi_splash.close(), 2)
            
        self.icon = resource_path('assets/icon.ico')
        request_android_permissions()
        self.title = "Favors CNN Weed Detector"
        model_path = resource_path("assets/weed_detector.tflite")
        self.detector = WeedDetector(model_path)
        self.interpreter = self.detector.interpreter
        self.input_details = self.detector.input_details
        self.output_details = self.detector.output_details
        self.labels = self.detector.classes
        # Set app icon
        self.icon = resource_path('assets/icon.ico')
        return Builder.load_string(KV)

    def close_splash(self, *args):
        self.splash.dismiss()

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
            self.root.ids.video_feed.texture = None  # Clear the video feed

            self.root.ids.start_camera_button.icon = 'camera'

    def update_camera(self, dt):
        if self.capture.isOpened() and self.camera_active:
            ret, frame = self.capture.read()
            if ret:
                # Resize and preprocess the frame for the model
                img = cv2.resize(frame, (128, 128))  # Model input size
                img = img.astype('float32') / 255.0  # Normalize
                input_data = np.expand_dims(img, axis=0)

                # Run the model
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

                # Get prediction
                predicted_index = np.argmax(output_data[0])
                predicted_label = self.labels[predicted_index]

                # Calculate position dynamically
                frame_height, frame_width, _ = frame.shape
                text_x = int(frame_width / 2)  # Center horizontally
                text_y = int(frame_height / 2) # Center vertically

                # Overlay prediction on the frame
                cv2.putText(frame, predicted_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Flip the frame vertically (to match Kivy's coordinate system)
                buf = cv2.flip(frame, 0).tobytes()

                # Create a texture from the frame
                image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                # Update the FitImage with the new texture
                self.root.ids.video_feed.texture = image_texture

    def pick_video_file(self):
        # File chooser to select a video file
        filechooser = FileChooserIconView(path=os.getcwd(), filters=["*.mp4"])
        popup = Popup(title="Select a Video", content=filechooser, size_hint=(0.9, 0.9))
        filechooser.bind(on_submit=lambda fc, selection, touch: self.load_selected_video(selection, popup))
        popup.open()

    def detect_from_video(self, video_path):
        # Detect from a selected video file
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

            # Same detection logic as for camera
            img = cv2.resize(frame, (128, 128))
            img = img.astype('float32') / 255.0
            input_data = np.expand_dims(img, axis=0)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            predicted_index = np.argmax(output_data[0])
            predicted_label = self.labels[predicted_index]

            # Modified text parameters
            font_scale = 3.0  # Increased from 1.0
            thickness = 4     # Increased from 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size to better position it
            (text_width, text_height), baseline = cv2.getTextSize(
                predicted_label, font, font_scale, thickness
            )
            
            # Calculate position dynamically
            frame_height, frame_width, _ = frame.shape
            text_x = int((frame_width - text_width) / 2)  # Center horizontally
            text_y = int((frame_height + text_height) / 2)  # Center vertically

            # Draw text with larger size
            cv2.putText(
                frame, 
                predicted_label, 
                (text_x, text_y), 
                font, 
                font_scale,
                (255, 0, 0),  # Blue color
                thickness,
                cv2.LINE_AA  # Anti-aliased line for better quality
            )
            out.write(frame)

            # Flip the frame vertically (to match Kivy's coordinate system)
            buf = cv2.flip(frame, 0).tobytes()

            # Create a texture from the frame
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # Update the FitImage with the new texture
            self.root.ids.video_feed.texture = image_texture

        Clock.schedule_interval(process_frame, 1.0 / 30.0)  # Schedule frame processing

    def load_selected_video(self, selection, popup):
        if selection:
            selected_path = selection[0]
            print(f"🎞️ Selected video: {selected_path}")
            self.detect_from_video(selected_path)

        popup.dismiss()

    def show_logs(self):
        # Placeholder for logs feature
        print("📜 Logs feature will be added soon...")

    def show_error_popup(self, title, message):
        popup = Popup(
            title=title,
            content=MDLabel(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()

    def on_stop(self):
        """Clean up resources when the application exits"""
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
        if hasattr(self, 'interpreter'):
            del self.interpreter
        cv2.destroyAllWindows()

    def check_opencv_config(self):
        """Check if OpenCV is properly configured"""
        try:
            # Try to get the first camera
            temp_cap = cv2.VideoCapture(0)
            if temp_cap is None or not temp_cap.isOpened():
                return False
            temp_cap.release()
            return True
        except Exception:
            return False

FavourBroadLeafDetectorApp().run()
