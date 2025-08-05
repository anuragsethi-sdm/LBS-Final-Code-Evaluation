import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from utils.helpers import preprocess_face

# Page configuration
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="",
    layout="wide"
)

st.title(" Face Mask Detection System")
st.markdown("Upload an image to detect faces and determine if they're wearing masks")

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model/face_mask_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load face cascade
@st.cache_resource
def load_face_cascade():
    try:
        # Try to load custom cascade first
        face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            # Fallback to OpenCV's built-in cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face cascade: {e}")
        return None

class FaceMaskDetector:
    def __init__(self, model, face_cascade):
        self.model = model
        self.face_cascade = face_cascade
        
    def detect_faces(self, image):
        """Detect faces with multiple parameter sets"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try different detection parameters
        detection_params = [
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (30, 30)},
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (50, 50)},
            {'scaleFactor': 1.2, 'minNeighbors': 4, 'minSize': (40, 40)},
            {'scaleFactor': 1.3, 'minNeighbors': 6, 'minSize': (60, 60)},
        ]
        
        for params in detection_params:
            faces = self.face_cascade.detectMultiScale(gray, **params)
            if len(faces) > 0:
                return faces
        
        return np.array([])
    
    def enhanced_mask_detection(self, face_region):
        """Enhanced mask detection using multiple methods"""
        scores = []
        
        # Method 1: ML Model prediction
        if self.model is not None:
            try:
                processed_face = preprocess_face(face_region)
                ml_score = self.model.predict(processed_face, verbose=0)[0][0]
                scores.append(ml_score * 0.4)
            except:
                scores.append(0.2)  # Default score if model fails
        else:
            scores.append(0.2)
        
        # Method 2: Color-based detection
        color_score = self.detect_mask_colors(face_region)
        scores.append(color_score * 0.3)
        
        # Method 3: Lower face analysis
        lower_face_score = self.analyze_lower_face(face_region)
        scores.append(lower_face_score * 0.3)
        
        return sum(scores)
    
    def detect_mask_colors(self, face_region):
        """Detect mask-like colors"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Define mask color ranges
        mask_ranges = [
            # Blue masks (surgical)
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # White masks
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # Black masks
            (np.array([0, 0, 0]), np.array([180, 255, 60])),
            # Gray masks
            (np.array([0, 0, 50]), np.array([180, 30, 200])),
        ]
        
        # Focus on lower half of face
        h = face_region.shape[0]
        lower_half = hsv[h//2:, :]
        total_pixels = lower_half.shape[0] * lower_half.shape[1]
        
        mask_pixels = 0
        for lower, upper in mask_ranges:
            mask = cv2.inRange(lower_half, lower, upper)
            mask_pixels += cv2.countNonZero(mask)
        
        return min(mask_pixels / total_pixels * 2, 1.0)
    
    def analyze_lower_face(self, face_region):
        """Analyze lower face for mask indicators"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Split face into upper and lower regions
        upper_region = gray[:h//2, :]
        lower_region = gray[h//2:, :]
        
        # Calculate texture differences
        upper_std = np.std(upper_region)
        lower_std = np.std(lower_region)
        
        # If lower region has different texture characteristics, might be masked
        if upper_std > 0:
            texture_diff = abs(lower_std - upper_std) / upper_std
            return min(texture_diff, 1.0)
        
        return 0.0
    
    def process_image(self, image):
        """Process image and detect masks"""
        # Convert PIL to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = self.detect_faces(image)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                continue
            
            # Detect mask
            mask_score = self.enhanced_mask_detection(face_region)
            
            # Determine label (adjusted threshold for better sensitivity)
            threshold = 0.25
            if mask_score > threshold:
                label = "Mask"
                confidence = min(mask_score * 100, 95)  # Cap confidence
                color = (0, 255, 0)  # Green
            else:
                label = "No Mask"
                confidence = min((1 - mask_score) * 100, 95)
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
        """Draw detection results"""
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
            
            # Background for text
            cv2.rectangle(result_image, (x, y-text_height-10), 
                         (x+text_width, y), color, -1)
            cv2.putText(result_image, text, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image

# Load components
model = load_model()
face_cascade = load_face_cascade()

if model is None or face_cascade is None:
    st.error("Failed to load required components. Please check the model files.")
    st.stop()

# Initialize detector
detector = FaceMaskDetector(model, face_cascade)

# Sidebar
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Detection Sensitivity", 0.1, 0.9, 0.25, 0.05)
show_debug_info = st.sidebar.checkbox("Show Debug Information", False)

# Main interface
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image containing faces to detect mask usage"
)

# Sample images section
st.sidebar.markdown("### Sample Images")
if st.sidebar.button("Create Sample Images"):
    # Create some sample images for testing
    sample_dir = "sample_images"
    import os
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a simple test image
    test_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
    
    # Draw a simple face
    cv2.circle(test_img, (200, 200), 80, (180, 160, 140), -1)  # Face
    cv2.circle(test_img, (170, 170), 10, (0, 0, 0), -1)        # Left eye
    cv2.circle(test_img, (230, 170), 10, (0, 0, 0), -1)        # Right eye
    cv2.circle(test_img, (200, 200), 5, (150, 120, 100), -1)   # Nose
    cv2.ellipse(test_img, (200, 230), (20, 10), 0, 0, 180, (120, 80, 80), -1)  # Mouth
    
    cv2.imwrite(f"{sample_dir}/no_mask_sample.jpg", test_img)
    
    # Create masked version
    masked_img = test_img.copy()
    cv2.rectangle(masked_img, (150, 210), (250, 260), (255, 255, 255), -1)  # White mask
    cv2.imwrite(f"{sample_dir}/mask_sample.jpg", masked_img)
    
    st.sidebar.success("Sample images created in sample_images folder!")

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Detection Results")
        
        with st.spinner("Analyzing image..."):
            # Process image
            results = detector.process_image(image)
            
            if results:
                # Convert back to RGB for display
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                result_image = detector.draw_results(opencv_image, results)
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                st.image(result_image_rgb, use_column_width=True)
                
                # Statistics
                st.subheader("Detection Summary")
                total_faces = len(results)
                mask_count = sum(1 for r in results if r['label'] == 'Mask')
                no_mask_count = total_faces - mask_count
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Total Faces", total_faces)
                with col_stats2:
                    st.metric("With Mask", mask_count)
                with col_stats3:
                    st.metric("Without Mask", no_mask_count)
                
                # Detailed results
                st.subheader("Detailed Results")
                for i, result in enumerate(results):
                    with st.expander(f"Face {i+1}: {result['label']}"):
                        st.write(f"**Label:** {result['label']}")
                        st.write(f"**Confidence:** {result['confidence']:.1f}%")
                        st.write(f"**Bounding Box:** {result['bbox']}")
                        
                        if show_debug_info:
                            st.write(f"**Raw Score:** {result['mask_score']:.3f}")
                            st.write(f"**Threshold Used:** {confidence_threshold}")
            else:
                st.warning("No faces detected in the image. Try uploading a different image with clear faces.")
                
                if show_debug_info:
                    st.write("**Debug Info:** Face detection failed. This could be due to:")
                    st.write("- Image quality or lighting")
                    st.write("- Face angle or size")
                    st.write("- Image resolution")

else:
    st.info(" Upload an image to get started!")
    
    # Instructions
    st.markdown("""
    ### How to use:
    1. **Upload an image** using the file uploader above
    2. **Wait for processing** - the system will detect faces and classify masks
    3. **View results** - see bounding boxes and confidence scores
    
    ### Features:
    -  **Face Detection**: Automatically finds faces in images
    -  **Mask Classification**: Determines if each person is wearing a mask
    -  **Confidence Scores**: Shows how certain the system is about each prediction
    -  **Multiple Detection Methods**: Combines ML model with color and texture analysis
    
    ### Tips for best results:
    - Use clear, well-lit images
    - Ensure faces are clearly visible
    - Avoid very small or blurry faces
    - Images with multiple people work too!
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, OpenCV, and TensorFlow")