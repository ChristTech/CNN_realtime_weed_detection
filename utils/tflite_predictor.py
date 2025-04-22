import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

class WeedDetector:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.classes = ['soil', 'broadleaf', 'grass', 'soybean']

    def predict(self, image: Image.Image):
        image = image.resize((128, 128))
        img_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return self.classes[np.argmax(output)], float(np.max(output))
