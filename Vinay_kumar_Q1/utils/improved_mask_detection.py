#!/usr/bin/env python3
"""
Improved Face Mask Detection with Real Image Testing
"""

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.helpers import preprocess_face

print(" Improved Face Mask Detection System")
print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# Load the trained model
try:
    model = tf.keras.models.load_model('model/face_mask_model.h5')
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    print("Please run the training script first.")
    exit(1)

# Load face cascade
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Using OpenCV's built-in face cascade")

class ImprovedFaceMaskDetector:
    def __init__(self, model, face_cascade):
        self.model = model
        self.face_cascade = face_cascade
        
    def detect_faces(self, image):
        """Improved face detection with multiple scales"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection parameters for better results
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller scale factor for better detection
            minNeighbors=3,    # Reduced neighbors for more detections
            minSize=(50, 50),  # Smaller minimum size
            maxSize=(300, 300) # Maximum size limit
        )
        
        # If no faces found, try with different parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(500, 500)
            )
        
        return faces
    
    def enhanced_mask_detection(self, face_region):
        """Enhanced mask detection combining ML model and rule-based approach"""
        # Method 1: Use trained ML model
        try:
            processed_face = preprocess_face(face_region)
            ml_prediction = self.model.predict(processed_face, verbose=0)[0][0]
        except:
            ml_prediction = 0.5  # Default if model fails
        
        # Method 2: Color-based detection
        color_score = self.detect_mask_colors(face_region)
        
        # Method 3: Edge-based detection
        edge_score = self.detect_mask_edges(face_region)
        
        # Method 4: Lower face coverage detection
        coverage_score = self.detect_lower_face_coverage(face_region)
        
        # Combine all methods with weights
        combined_score = (
            ml_prediction * 0.4 +      # ML model weight
            color_score * 0.25 +       # Color detection weight
            edge_score * 0.2 +         # Edge detection weight
            coverage_score * 0.15      # Coverage detection weight
        )
        
        return combined_score
    
    def detect_mask_colors(self, face_region):
        """Detect mask-like colors in face region"""
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Enhanced mask color ranges
        mask_color_ranges = [
            # Blue surgical masks (wider range)
            (np.array([90, 50, 50]), np.array([130, 255, 255])),
            # White/light masks (adjusted)
            (np.array([0, 0, 180]), np.array([180, 40, 255])),
            # Black/dark masks
            (np.array([0, 0, 0]), np.array([180, 255, 80])),
            # Green masks
            (np.array([35, 40, 40]), np.array([85, 255, 255])),
            # Gray masks
            (np.array([0, 0, 80]), np.array([180, 40, 180])),
        ]
        
        total_pixels = face_region.shape[0] * face_region.shape[1]
        mask_pixels = 0
        
        # Focus on lower half of face where mask would be
        lower_half = hsv[hsv.shape[0]//2:, :]
        lower_pixels = lower_half.shape[0] * lower_half.shape[1]
        
        for lower, upper in mask_color_ranges:
            mask = cv2.inRange(lower_half, lower, upper)
            mask_pixels += cv2.countNonZero(mask)
        
        # Calculate ratio based on lower face area
        mask_ratio = min(mask_pixels / lower_pixels, 1.0)
        return mask_ratio
    
    def detect_mask_edges(self, face_region):
        """Detect mask edges in lower face area"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Focus on lower 60% of face
        h = face_region.shape[0]
        lower_region = edges[int(h*0.4):, :]
        
        # Count edge pixels
        edge_pixels = np.sum(lower_region > 0)
        total_pixels = lower_region.shape[0] * lower_region.shape[1]
        
        edge_density = edge_pixels / total_pixels
        
        # Masks typically create more defined edges in lower face
        return min(edge_density * 5, 1.0)  # Scale up the score
    
    def detect_lower_face_coverage(self, face_region):
        """Detect if lower face area is covered/obscured"""
        # Convert to LAB color space for better analysis
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        
        # Split into upper and lower halves
        h = face_region.shape[0]
        upper_half = lab[:h//2, :]
        lower_half = lab[h//2:, :]
        
        # Calculate color variance in each half
        upper_var = np.var(upper_half)
        lower_var = np.var(lower_half)
        
        # If lower half has significantly different characteristics, likely masked
        if upper_var > 0:
            variance_ratio = abs(lower_var - upper_var) / upper_var
            return min(variance_ratio, 1.0)
        
        return 0.0
    
    def process_image(self, image_path):
        """Process image with improved detection"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            image = image_path
        
        # Detect faces
        faces = self.detect_faces(image)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region with some padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                continue
            
            # Enhanced mask detection
            mask_score = self.enhanced_mask_detection(face_region)
            
            # Determine label with adjusted threshold
            threshold = 0.3  # Lower threshold for better sensitivity
            if mask_score > threshold:
                label = "Mask"
                confidence = mask_score * 100
                color = (0, 255, 0)  # Green
            else:
                label = "No Mask"
                confidence = (1 - mask_score) * 100
                color = (0, 0, 255)  # Red
            
            results.append({
                'bbox': (x, y, w, h),
                'label': label,
                'confidence': confidence,
                'color': color,
                'mask_score': mask_score
            })
        
        return results
    
    def draw_results(self, image, results):
        """Draw detection results on image"""
        result_image = image.copy()
        
        for result in results:
            x, y, w, h = result['bbox']
            label = result['label']
            confidence = result['confidence']
            color = result['color']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 3)
            
            # Draw label with background
            text = f"{label}: {confidence:.1f}%"
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Background rectangle
            cv2.rectangle(result_image, (x, y-text_height-15), 
                         (x+text_width+10, y), color, -1)
            
            # Text
            cv2.putText(result_image, text, (x+5, y-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return result_image

# Initialize improved detector
detector = ImprovedFaceMaskDetector(model, face_cascade)
print(" Improved Face Mask Detector initialized!")

def create_realistic_test_images():
    """Create more realistic test images"""
    os.makedirs('test_images', exist_ok=True)
    
    test_cases = []
    
    for i in range(6):
        # Create larger, more realistic face image
        img = np.ones((500, 400, 3), dtype=np.uint8) * 220  # Light background
        
        # Face oval (more realistic proportions)
        center_x, center_y = 200, 200
        face_width, face_height = 120, 160
        
        # Skin color face
        cv2.ellipse(img, (center_x, center_y), (face_width, face_height), 
                   0, 0, 360, (180, 160, 140), -1)
        
        # Eyes (more detailed)
        eye_y = center_y - 40
        # Left eye
        cv2.ellipse(img, (center_x-40, eye_y), (20, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (center_x-40, eye_y), 8, (100, 50, 20), -1)
        cv2.circle(img, (center_x-40, eye_y), 4, (0, 0, 0), -1)
        
        # Right eye
        cv2.ellipse(img, (center_x+40, eye_y), (20, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (center_x+40, eye_y), 8, (100, 50, 20), -1)
        cv2.circle(img, (center_x+40, eye_y), 4, (0, 0, 0), -1)
        
        # Eyebrows
        cv2.ellipse(img, (center_x-40, eye_y-20), (25, 8), 0, 0, 180, (80, 60, 40), -1)
        cv2.ellipse(img, (center_x+40, eye_y-20), (25, 8), 0, 0, 180, (80, 60, 40), -1)
        
        # Nose
        nose_points = np.array([
            [center_x-8, center_y-10],
            [center_x+8, center_y-10],
            [center_x+12, center_y+10],
            [center_x-12, center_y+10]
        ], np.int32)
        cv2.fillPoly(img, [nose_points], (160, 140, 120))
        
        # Nostrils
        cv2.circle(img, (center_x-6, center_y+8), 2, (120, 100, 80), -1)
        cv2.circle(img, (center_x+6, center_y+8), 2, (120, 100, 80), -1)
        
        # Mouth
        cv2.ellipse(img, (center_x, center_y+50), (25, 8), 0, 0, 180, (120, 80, 80), -1)
        
        # Determine if this face should have a mask
        has_mask = i % 2 == 0  # Alternate mask/no mask
        
        if has_mask:
            # Add realistic mask
            mask_colors = [
                (240, 240, 240),  # White surgical mask
                (100, 150, 255),  # Blue surgical mask
                (60, 60, 60),     # Black cloth mask
            ]
            mask_color = mask_colors[i // 2]
            
            # Main mask area (covers nose and mouth)
            mask_points = np.array([
                [center_x-70, center_y+10],   # Left side
                [center_x+70, center_y+10],   # Right side
                [center_x+60, center_y+80],   # Bottom right
                [center_x-60, center_y+80],   # Bottom left
            ], np.int32)
            cv2.fillPoly(img, [mask_points], mask_color)
            
            # Mask straps
            cv2.line(img, (center_x-70, center_y+30), (center_x-120, center_y-20), mask_color, 8)
            cv2.line(img, (center_x+70, center_y+30), (center_x+120, center_y-20), mask_color, 8)
            
            # Add mask texture/pleats
            for j in range(3):
                y_line = center_y + 25 + j * 15
                cv2.line(img, (center_x-60, y_line), (center_x+60, y_line), 
                        (mask_color[0]//2, mask_color[1]//2, mask_color[2]//2), 1)
            
            # Add some shadow under mask
            shadow_points = np.array([
                [center_x-65, center_y+75],
                [center_x+65, center_y+75],
                [center_x+55, center_y+85],
                [center_x-55, center_y+85],
            ], np.int32)
            cv2.fillPoly(img, [shadow_points], (mask_color[0]//2, mask_color[1]//2, mask_color[2]//2))
        
        # Add some hair
        cv2.ellipse(img, (center_x, center_y-80), (face_width+20, 60), 
                   0, 0, 180, (60, 40, 20), -1)
        
        # Save image
        filename = f'test_images/realistic_face_{i+1}.jpg'
        cv2.imwrite(filename, img)
        
        test_cases.append((filename, has_mask))
        print(f"  Created {filename} - Expected: {'Mask' if has_mask else 'No Mask'}")
    
    return test_cases

# Create realistic test images
print("\n Creating Realistic Test Images...")
test_cases = create_realistic_test_images()

# Test the improved detector
print("\n Testing Improved Detection...")
correct_predictions = 0
total_predictions = 0

for filename, expected_mask in test_cases:
    print(f"\n Testing {filename}:")
    print(f"   Expected: {'Mask' if expected_mask else 'No Mask'}")
    
    # Load and process image
    image = cv2.imread(filename)
    results = detector.process_image(image)
    
    if results:
        for i, result in enumerate(results):
            predicted_mask = result['label'] == 'Mask'
            is_correct = predicted_mask == expected_mask
            correct_predictions += is_correct
            total_predictions += 1
            
            status = " CORRECT" if is_correct else " INCORRECT"
            print(f"   Face {i+1}: {result['label']} ({result['confidence']:.1f}%) - Score: {result['mask_score']:.3f} {status}")
            
            # Draw results and save
            result_image = detector.draw_results(image, results)
            result_filename = filename.replace('.jpg', '_result.jpg')
            cv2.imwrite(result_filename, result_image)
    else:
        print("    No faces detected")

# Calculate and display final accuracy
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n Final Test Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
else:
    print("\n No faces were detected in any test images")

print("\n Results saved with '_result.jpg' suffix")
print(" Improved detection testing completed!")

# Test with webcam (optional)
def test_webcam():
    """Test with webcam"""
    print("\n📹 Starting webcam test (press 'q' to quit)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = detector.process_image(frame)
        
        # Draw results
        if results:
            frame = detector.draw_results(frame, results)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Improved Face Mask Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test completed!")

# Uncomment to test with webcam
# test_webcam()

print("\n All testing completed! Check the result images to see detection performance.")