#!/usr/bin/env python3
"""
Quick Face Mask Detection Training and Prediction Script
Runs the complete pipeline: data generation -> training -> prediction -> results
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from utils.helpers import preprocess_face, detect_mask_colors
import time

print(" Starting Face Mask Detection Pipeline...")
print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# ============================================================================
# STEP 1: DATA GENERATION
# ============================================================================
print("\n📊 STEP 1: Generating Training Data...")

def create_synthetic_data(num_samples=1000):
    """Create synthetic face mask data for training"""
    X_data = []
    y_data = []
    
    print(f"Creating {num_samples} synthetic samples...")
    
    for i in range(num_samples):
        if i % 200 == 0:
            print(f"  Generated {i}/{num_samples} samples...")
        
        # Create random face-like image
        img = np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
        
        # Add face features
        # Eyes
        cv2.circle(img, (50, 50), 10, (0, 0, 0), -1)
        cv2.circle(img, (100, 50), 10, (0, 0, 0), -1)
        cv2.circle(img, (50, 50), 5, (255, 255, 255), -1)
        cv2.circle(img, (100, 50), 5, (255, 255, 255), -1)
        
        # Nose
        cv2.circle(img, (75, 75), 5, (150, 100, 100), -1)
        
        # Mouth
        cv2.ellipse(img, (75, 100), (15, 8), 0, 0, 180, (100, 50, 50), -1)
        
        # Decide if this face has a mask (50% probability)
        has_mask = np.random.choice([0, 1])
        
        if has_mask:
            # Add mask-like rectangle in lower face area
            mask_colors = [
                [255, 255, 255],  # White mask
                [100, 150, 255],  # Blue mask
                [50, 50, 50],     # Black mask
                [200, 200, 200],  # Light gray mask
            ]
            mask_color = mask_colors[np.random.randint(0, len(mask_colors))]
            
            # Main mask area
            cv2.rectangle(img, (30, 90), (120, 130), mask_color, -1)
            
            # Add mask straps
            cv2.line(img, (30, 100), (10, 80), mask_color, 2)
            cv2.line(img, (120, 100), (140, 80), mask_color, 2)
            
            # Add some texture to mask
            for _ in range(5):
                x = np.random.randint(35, 115)
                y = np.random.randint(95, 125)
                cv2.circle(img, (x, y), 1, (mask_color[0]//2, mask_color[1]//2, mask_color[2]//2), -1)
        
        # Normalize and add to dataset
        img_normalized = img / 255.0
        X_data.append(img_normalized)
        y_data.append(has_mask)
    
    return np.array(X_data), np.array(y_data)

# Generate data
X, y = create_synthetic_data(2000)
print(f"✅ Created {len(X)} samples")
print(f"   Data shape: {X.shape}")
print(f"   Mask samples: {np.sum(y)}, No mask samples: {len(y) - np.sum(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 2: MODEL CREATION AND TRAINING
# ============================================================================
print("\n🧠 STEP 2: Building and Training Model...")

def create_mask_detection_model():
    """Create CNN model for mask detection"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Fourth Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten and Dense layers
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    return model

# Create and compile model
model = create_mask_detection_model()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model Architecture:")
model.summary()

# Train the model
print("\n🏋️ Training the model...")
start_time = time.time()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Train with reduced epochs for quick execution
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=10,  # Reduced for quick execution
    validation_data=(X_test, y_test),
    verbose=1
)

training_time = time.time() - start_time
print(f"✅ Training completed in {training_time:.2f} seconds!")

# ============================================================================
# STEP 3: MODEL EVALUATION
# ============================================================================
print("\n📈 STEP 3: Evaluating Model...")

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions
predictions = model.predict(X_test, verbose=0)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes, target_names=['No Mask', 'Mask']))

# Confusion Matrix
cm = confusion_matrix(y_test, predicted_classes)
print(f"\nConfusion Matrix:")
print(f"True Negatives (No Mask correctly predicted): {cm[0,0]}")
print(f"False Positives (No Mask predicted as Mask): {cm[0,1]}")
print(f"False Negatives (Mask predicted as No Mask): {cm[1,0]}")
print(f"True Positives (Mask correctly predicted): {cm[1,1]}")

# ============================================================================
# STEP 4: SAVE MODEL
# ============================================================================
print("\n💾 STEP 4: Saving Model...")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save model
model.save('model/face_mask_model.h5')
print("✅ Model saved as 'model/face_mask_model.h5'")

# Save in SavedModel format for deployment
model.save('model/face_mask_model_savedmodel')
print("✅ Model saved in SavedModel format for deployment")

# ============================================================================
# STEP 5: FACE DETECTION AND PREDICTION PIPELINE
# ============================================================================
print("\n🔍 STEP 5: Setting up Prediction Pipeline...")

class FaceMaskDetector:
    def __init__(self, model, face_cascade_path=None):
        self.model = model
        
        # Load face cascade
        if face_cascade_path and os.path.exists(face_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        else:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        if self.face_cascade.empty():
            print("⚠️ Warning: Could not load face cascade classifier")
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def predict_mask(self, face_region):
        """Predict if face has mask"""
        processed_face = preprocess_face(face_region)
        prediction = self.model.predict(processed_face, verbose=0)
        return prediction[0][0]
    
    def process_image(self, image):
        """Process image and detect masks"""
        faces = self.detect_faces(image)
        results = []
        
        for (x, y, w, h) in faces:
            face_region = image[y:y+h, x:x+w]
            mask_prob = self.predict_mask(face_region)
            
            if mask_prob > 0.5:
                label = "Mask"
                confidence = mask_prob * 100
                color = (0, 255, 0)  # Green
            else:
                label = "No Mask"
                confidence = (1 - mask_prob) * 100
                color = (0, 0, 255)  # Red
            
            results.append({
                'bbox': (x, y, w, h),
                'label': label,
                'confidence': confidence,
                'color': color,
                'mask_probability': mask_prob
            })
        
        return results
    
    def draw_results(self, image, results):
        """Draw bounding boxes and labels"""
        result_image = image.copy()
        
        for result in results:
            x, y, w, h = result['bbox']
            label = result['label']
            confidence = result['confidence']
            color = result['color']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            text = f"{label}: {confidence:.1f}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(result_image, (x, y-text_height-10), 
                         (x+text_width, y), color, -1)
            cv2.putText(result_image, text, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image

# Initialize detector
detector = FaceMaskDetector(model, 'model/haarcascade_frontalface_default.xml')
print("✅ Face Mask Detector initialized!")

# ============================================================================
# STEP 6: CREATE AND TEST SAMPLE IMAGES
# ============================================================================
print("\n🖼️ STEP 6: Creating and Testing Sample Images...")

def create_test_images():
    """Create sample test images"""
    os.makedirs('test_images', exist_ok=True)
    
    test_results = []
    
    for i in range(4):
        # Create a face-like image
        img = np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
        
        # Add background
        img[:] = [180, 160, 140]  # Skin-like background
        
        # Face outline (oval)
        cv2.ellipse(img, (200, 200), (100, 130), 0, 0, 360, (200, 180, 160), -1)
        
        # Eyes
        cv2.circle(img, (170, 170), 15, (50, 50, 50), -1)
        cv2.circle(img, (230, 170), 15, (50, 50, 50), -1)
        cv2.circle(img, (170, 170), 8, (255, 255, 255), -1)
        cv2.circle(img, (230, 170), 8, (255, 255, 255), -1)
        cv2.circle(img, (170, 170), 4, (0, 0, 0), -1)
        cv2.circle(img, (230, 170), 4, (0, 0, 0), -1)
        
        # Eyebrows
        cv2.ellipse(img, (170, 150), (20, 8), 0, 0, 180, (100, 80, 60), -1)
        cv2.ellipse(img, (230, 150), (20, 8), 0, 0, 180, (100, 80, 60), -1)
        
        # Nose
        cv2.circle(img, (200, 200), 8, (180, 150, 130), -1)
        cv2.ellipse(img, (195, 205), (3, 2), 0, 0, 360, (120, 100, 80), -1)
        cv2.ellipse(img, (205, 205), (3, 2), 0, 0, 360, (120, 100, 80), -1)
        
        # Mouth
        cv2.ellipse(img, (200, 240), (20, 10), 0, 0, 180, (150, 100, 100), -1)
        
        # Add mask to some images
        has_mask = i % 2 == 0  # Alternate mask/no mask
        
        if has_mask:
            mask_colors = [
                (255, 255, 255),  # White
                (100, 150, 255),  # Blue
                (50, 50, 50),     # Black
                (200, 220, 200),  # Light green
            ]
            mask_color = mask_colors[i // 2]
            
            # Main mask area (covers nose and mouth)
            cv2.rectangle(img, (130, 190), (270, 260), mask_color, -1)
            
            # Mask straps
            cv2.line(img, (130, 210), (100, 180), mask_color, 4)
            cv2.line(img, (270, 210), (300, 180), mask_color, 4)
            
            # Add some mask details
            cv2.line(img, (140, 200), (260, 200), (mask_color[0]//2, mask_color[1]//2, mask_color[2]//2), 1)
            cv2.line(img, (140, 220), (260, 220), (mask_color[0]//2, mask_color[1]//2, mask_color[2]//2), 1)
            cv2.line(img, (140, 240), (260, 240), (mask_color[0]//2, mask_color[1]//2, mask_color[2]//2), 1)
        
        # Save image
        filename = f'test_images/test_face_{i+1}.jpg'
        cv2.imwrite(filename, img)
        
        # Test detection
        results = detector.process_image(img)
        test_results.append((filename, results, has_mask))
        
        print(f"  Created {filename} - Expected: {'Mask' if has_mask else 'No Mask'}")
    
    return test_results

# Create and test images
test_results = create_test_images()

# ============================================================================
# STEP 7: DISPLAY RESULTS
# ============================================================================
print("\n🎯 STEP 7: Testing Results...")

correct_predictions = 0
total_predictions = 0

for filename, results, expected_mask in test_results:
    print(f"\n📸 {filename}:")
    print(f"   Expected: {'Mask' if expected_mask else 'No Mask'}")
    
    if results:
        for i, result in enumerate(results):
            predicted_mask = result['label'] == 'Mask'
            is_correct = predicted_mask == expected_mask
            correct_predictions += is_correct
            total_predictions += 1
            
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"   Face {i+1}: {result['label']} ({result['confidence']:.1f}%) {status}")
    else:
        print("   No faces detected")

if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n🎯 Overall Test Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")

# ============================================================================
# STEP 8: PERFORMANCE BENCHMARKING
# ============================================================================
print("\n⚡ STEP 8: Performance Benchmarking...")

def benchmark_model():
    """Benchmark model performance"""
    print("Running performance benchmark...")
    
    # Create test batch
    test_batch = np.random.random((16, 150, 150, 3)).astype(np.float32)
    
    # Warm up
    for _ in range(3):
        _ = model.predict(test_batch, verbose=0)
    
    # Benchmark
    times = []
    for _ in range(10):
        start_time = time.time()
        _ = model.predict(test_batch, verbose=0)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 16 / avg_time
    per_image_ms = (avg_time / 16) * 1000
    
    print(f"   Average batch time: {avg_time:.4f} seconds")
    print(f"   Throughput: {fps:.2f} images/second")
    print(f"   Per image: {per_image_ms:.2f} ms")

benchmark_model()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*60)
print("🎉 FACE MASK DETECTION PIPELINE COMPLETED!")
print("="*60)
print(f"✅ Data Generation: {len(X)} samples created")
print(f"✅ Model Training: {training_time:.2f} seconds")
print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print(f"✅ Model Saved: model/face_mask_model.h5")
print(f"✅ Sample Images: 4 test images created and tested")
print(f"✅ Pipeline Ready: Use streamlit_app.py for web interface")
print("="*60)

print("\n🚀 Next Steps:")
print("1. Run Streamlit app: streamlit run streamlit_app.py")
print("2. Open Jupyter notebooks for detailed analysis")
print("3. Use the trained model for real-time detection")
print("4. Deploy using the provided deployment scripts")

print("\n📁 Generated Files:")
print("- model/face_mask_model.h5 (Trained model)")
print("- model/face_mask_model_savedmodel/ (Deployment format)")
print("- test_images/ (Sample test images)")

print("\n🎯 Model Performance Summary:")
print(f"- Training Samples: {len(X_train)}")
print(f"- Test Samples: {len(X_test)}")
print(f"- Final Accuracy: {test_accuracy:.4f}")
print(f"- Training Time: {training_time:.2f} seconds")

print("\n✨ Ready for deployment and real-world usage!")