import cv2
import numpy as np
import os
from PIL import Image
import logging
import face_recognition
import pickle

class FaceRecognizer:
    def __init__(self, data_path="data/known_faces"):
        self.data_path = data_path
        self.face_encodings = {}
        self.face_names = {}
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        # Load existing encodings if available
        self.load_encodings()

    def load_encodings(self):
        """Load existing face encodings from disk."""
        encoding_file = os.path.join(self.data_path, "encodings.pkl")
        if os.path.exists(encoding_file):
            try:
                with open(encoding_file, 'rb') as f:
                    data = pickle.load(f)
                    self.face_encodings = data.get('encodings', {})
                    self.face_names = data.get('names', {})
            except Exception as e:
                logging.error(f"Error loading encodings: {str(e)}")

    def save_encodings(self):
        """Save face encodings to disk."""
        encoding_file = os.path.join(self.data_path, "encodings.pkl")
        try:
            with open(encoding_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.face_encodings,
                    'names': self.face_names
                }, f)
        except Exception as e:
            logging.error(f"Error saving encodings: {str(e)}")

    def detect_face(self, img):
        """Detect face in an image and return the face locations."""
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get face locations
        face_locations = face_recognition.face_locations(rgb_img)
        
        if len(face_locations) == 0:
            return None, None
        
        # Return the first detected face and its location
        return rgb_img, face_locations[0]

    def save_training_image(self, image_data, employee_id):
        """Save a face image for training."""
        employee_dir = os.path.join(self.data_path, str(employee_id))
        
        if not os.path.exists(employee_dir):
            os.makedirs(employee_dir)

        # Detect face
        rgb_img, face_location = self.detect_face(image_data)
        if face_location is None:
            return False

        # Get face encoding
        face_encoding = face_recognition.face_encodings(rgb_img, [face_location])[0]
        
        # Save encoding and update name mapping
        self.face_encodings[employee_id] = face_encoding
        self.face_names[employee_id] = str(employee_id)
        
        # Save to disk
        self.save_encodings()

        # Save face image
        image_count = len([f for f in os.listdir(employee_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        filename = f"face_{image_count + 1}.jpg"
        image_path = os.path.join(employee_dir, filename)

        # Extract and save face region
        top, right, bottom, left = face_location
        face_image = image_data[top:bottom, left:right]
        cv2.imwrite(image_path, face_image)
        
        return True

    def recognize(self, image_data):
        """Recognize a face in the given image."""
        if not self.face_encodings:
            logging.error("No face encodings available")
            return None, None

        # Detect face
        rgb_img, face_location = self.detect_face(image_data)
        if face_location is None:
            return None, None

        # Get face encoding
        face_encoding = face_recognition.face_encodings(rgb_img, [face_location])[0]
        
        # Compare with known faces
        min_distance = float('inf')
        recognized_id = None
        
        for employee_id, known_encoding in self.face_encodings.items():
            # Compare face encodings
            distance = np.linalg.norm(face_encoding - known_encoding)
            
            # Update if this is the closest match
            if distance < min_distance and distance < 0.6:  # Threshold for recognition
                min_distance = distance
                recognized_id = employee_id

        if recognized_id:
            confidence = 1 - min_distance  # Convert distance to confidence score
            return recognized_id, confidence
        
        return None, None

    def get_face_encoding(self, image_data):
        """Get face encoding for an image."""
        rgb_img, face_location = self.detect_face(image_data)
        if face_location is None:
            return None
            
        face_encodings = face_recognition.face_encodings(rgb_img, [face_location])
        if not face_encodings:
            return None
            
        return face_encodings[0]
