import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import mediapipe as mp
import os
import time
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import uuid
import logging
import pathlib

# Configure logging
logging.basicConfig(
    filename="emotion_detection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

st.set_page_config(page_title="Real-Time Emotion Detection with 3D View", layout="wide")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("Model Path", r"D:\BINs\friends\FER\models\fer_model.pth")
    data_dir = st.text_input("Data Save Directory", r"D:\BINs\friends\FER\data\pictures")
    device_option = st.radio("Select Device", ["GPU (0)", "CPU (-1)"])
    adapt_training = st.checkbox("Enable Adaptive Training", value=False)
    max_frames = st.slider("Max Frames in Adaptive Mode", 2, 3, 3)
    st.button("🔄 Reset Session", key="reset_session", on_click=lambda: st.session_state.clear())

# Main UI
st.title("🎯 Real-Time Emotion Detection with 3D Face View")
st.markdown("Live 2D detection by default. Enable adaptive training for 2D/3D tabs and 2-3 frame capture with auto-stop.")

# Validate inputs
if not pathlib.Path(model_path).is_file():
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()
if not pathlib.Path(data_dir).is_dir():
    st.error(f"❌ Data directory not found: {data_dir}")
    st.stop()

device = torch.device("cuda:0" if "GPU" in device_option and torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define the CNN model
class FERModel(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(FERModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load the model
try:
    model = FERModel(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    st.success(f"✅ Model loaded from `{model_path}`")
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    logger.error(f"Model loading failed: {str(e)}")
    st.stop()

CLASS_NAMES = ['Boredom', 'Confused', 'Engaged', 'Frustration']

# Preprocessing function
def preprocess(face_img):
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(face_img).unsqueeze(0)
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return None

# Initialize MediaPipe Face Mesh
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    logger.info("MediaPipe Face Mesh initialized")
except Exception as e:
    st.error(f"❌ Failed to initialize MediaPipe: {str(e)}")
    logger.error(f"MediaPipe initialization failed: {str(e)}")
    st.stop()

# Create data directories
try:
    for emotion in CLASS_NAMES:
        os.makedirs(os.path.join(data_dir, emotion), exist_ok=True)
    logger.info(f"Data directories created at {data_dir}")
except Exception as e:
    st.error(f"❌ Failed to create data directories: {str(e)}")
    logger.error(f"Data directory creation failed: {str(e)}")
    st.stop()

# Temporal smoothing
emotion_history = deque(maxlen=10)
last_emotion = None

# Adaptive training setup
if adapt_training:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    st.markdown(f"🧠 **Adaptive Training Enabled**: Captures up to {max_frames} frames with 2D/3D tabs, then stops.")
    logger.info("Adaptive training enabled")

# Streamlit UI elements
if adapt_training:
    tab1, tab2 = st.tabs(["2D Video Feed", "3D Face Mesh"])
    frame_window = tab1.image([])
else:
    frame_window = st.image([])

result_placeholder = st.empty()
dimension_placeholder = st.empty()
adapt_frame_placeholder = st.empty()
save_frame_button = st.empty()
status_bar = st.empty()

# 3D plot function
@st.cache_data
def plot_3d_face(landmarks, frame_shape, _cache_key):
    try:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        h, w = frame_shape[:2]
        x = [landmark.x * w for landmark in landmarks.landmark]
        y = [landmark.y * h for landmark in landmarks.landmark]
        z = [landmark.z * w for landmark in landmarks.landmark]
        ax.scatter(x, y, z, s=1, c='b')
        connections = mp_face_mesh.FACEMESH_TESSELATION
        for conn in connections:
            start, end = conn
            ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'g', linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=10, azim=45)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        logger.error(f"3D plot error: {str(e)}")
        return None

if st.button("🎥 Start Camera + Detect", key="start_camera"):
    with st.spinner("Initializing camera..."):
        # Reset session state
        if "stop" not in st.session_state:
            st.session_state.stop = False
        if "frame_count" not in st.session_state:
            st.session_state.frame_count = 0
        if "transition_count" not in st.session_state:
            st.session_state.transition_count = 0
        if "last_frame_time" not in st.session_state:
            st.session_state.last_frame_time = time.time()

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Camera not accessible")
            logger.info("Camera initialized")
        except Exception as e:
            st.error(f"❌ Failed to open camera: {str(e)}")
            logger.error(f"Camera initialization failed: {str(e)}")
            st.stop()

        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                raise Exception("Haar cascade classifier not loaded")
            logger.info("Face cascade classifier loaded")
        except Exception as e:
            st.error(f"❌ Failed to load Haar cascade classifier: {str(e)}")
            logger.error(f"Classifier loading failed: {str(e)}")
            cap.release()
            st.stop()

    st.markdown("✅ **Camera Started. Detecting emotions...**")
    fig_3d_placeholder = tab2.empty() if adapt_training else None
    progress_bar = st.progress(0) if adapt_training else None

    while not st.session_state.stop:
        current_time = time.time()
        if current_time - st.session_state.last_frame_time < 1/30:  # 30 FPS cap
            continue
        st.session_state.last_frame_time = current_time
        fps = 1 / (current_time - st.session_state.last_frame_time + 1e-6)

        try:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to grab frame")
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        except Exception as e:
            st.error(f"❌ Frame processing error: {str(e)}")
            logger.error(f"Frame processing failed: {str(e)}")
            break

        # Process 3D face mesh
        face_dimensions = "No face detected"
        if adapt_training:
            try:
                mesh_results = face_mesh.process(rgb_frame)
                if mesh_results.multi_face_landmarks:
                    for landmarks in mesh_results.multi_face_landmarks:
                        h, w = frame.shape[:2]
                        nose_tip = landmarks.landmark[4]
                        chin = landmarks.landmark[152]
                        nose_to_chin_px = abs(nose_tip.y * h - chin.y * h)
                        face_width_px = abs(landmarks.landmark[234].x * w - landmarks.landmark[454].x * w)
                        face_depth = abs(landmarks.landmark[4].z * w - landmarks.landmark[152].z * w)
                        face_dimensions = f"Width: {face_width_px:.1f}px, Height: {nose_to_chin_px:.1f}px, Depth: {face_depth:.1f}px"
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_spec
                        )
                        cache_key = hash(str([(l.x, l.y, l.z) for l in landmarks.landmark]))
                        fig_3d = plot_3d_face(landmarks, frame.shape, _cache_key=cache_key)
                        if fig_3d:
                            fig_3d_placeholder.image(fig_3d, caption="3D Face Mesh")
            except Exception as e:
                logger.error(f"3D mesh processing error: {str(e)}")

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            input_tensor = preprocess(face_img)
            if input_tensor is None:
                continue
            input_tensor = input_tensor.to(device)
            try:
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    predicted_class = CLASS_NAMES[torch.argmax(output, dim=1).item()]
            except Exception as e:
                logger.error(f"Model inference error: {str(e)}")
                continue

            # Weighted temporal smoothing
            emotion_history.append((predicted_class, probs[CLASS_NAMES.index(predicted_class)]))
            if len(emotion_history) > 5:
                weighted_counts = {}
                for cls in CLASS_NAMES:
                    weighted_sum = sum(prob for c, prob in emotion_history if c == cls)
                    weighted_counts[cls] = weighted_sum
                smoothed_class = max(weighted_counts, key=weighted_counts.get)
            else:
                smoothed_class = predicted_class

            if smoothed_class != last_emotion and last_emotion is not None:
                st.session_state.transition_count += 1
                print(f"Emotion Transition #{st.session_state.transition_count}: {last_emotion} -> {smoothed_class}")
                logger.info(f"Emotion transition: {last_emotion} -> {smoothed_class}")
            last_emotion = smoothed_class

            # Draw face rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 50, 50), 2)
            cv2.putText(frame, smoothed_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display results
            result_placeholder.markdown(f"### 🧠 Detected Emotion: `{smoothed_class}` (Confidence: {probs[CLASS_NAMES.index(smoothed_class)]:.2%})")
            dimension_placeholder.markdown(f"### 📏 Face Dimensions only for training else 0 : {face_dimensions}")

            # Save frame in adaptive mode
            if adapt_training:
                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    frame_path = os.path.join(data_dir, smoothed_class, f"{timestamp}.jpg")
                    cv2.imwrite(frame_path, frame)
                    st.session_state.frame_count += 1
                    logger.info(f"Frame saved: {frame_path}")
                    if progress_bar:
                        progress_bar.progress(min(st.session_state.frame_count / max_frames, 1.0))

                    # Adaptive training
                    unique_key = f"train_{timestamp}_{uuid.uuid4().hex}"
                    if save_frame_button.button(f"Use Frame for {smoothed_class} Training", key=unique_key):
                        label = CLASS_NAMES.index(smoothed_class)
                        label_tensor = torch.tensor([label], dtype=torch.long).to(device)
                        optimizer.zero_grad()
                        output = model(input_tensor)
                        loss = criterion(output, label_tensor)
                        loss.backward()
                        optimizer.step()
                        adapt_frame_placeholder.markdown(f"✅ Frame used to fine-tune model for `{smoothed_class}`.")
                        logger.info(f"Model fine-tuned for {smoothed_class}")

                    # Stop after max_frames
                    if st.session_state.frame_count >= max_frames:
                        st.session_state.stop = True
                        st.markdown(f"✅ **{max_frames} frames captured and camera stopped.**")
                        logger.info(f"Captured {max_frames} frames, stopping camera")
                        break
                except Exception as e:
                    st.error(f"❌ Failed to save frame: {str(e)}")
                    logger.error(f"Frame saving failed: {str(e)}")

        frame_window.image(rgb_frame)
        status_bar.markdown(f"📸 Camera check Status : Camera Active | FPS: {fps:.1f} | Frames Captured: {st.session_state.frame_count}")

        # Stop button
        stop_key = f"stop_camera_{uuid.uuid4().hex}"
        if "stop_displayed" not in st.session_state:
            st.session_state.stop_displayed = False
        if not st.session_state.stop_displayed:
            if st.button("🛑 Stop Camera", key="stop_camera_button"):
                st.session_state.stop = True
                st.session_state.frame_count = 0
                st.session_state.transition_count = 0
                st.markdown("🛑 **Camera Stopped. Session reset.**")
            st.session_state.stop_displayed = True

    try:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released")
    except Exception as e:
        logger.error(f"Camera release error: {str(e)}")