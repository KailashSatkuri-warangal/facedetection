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

st.set_page_config(page_title="Real-Time Emotion Detection with 3D View", layout="centered")

st.title("🎯 Real-Time Emotion Detection with 3D Face View")
st.markdown("Click **Start Camera + Detect** to track facial emotions, view 3D face mesh, and enable adaptive training.")

# Input fields
model_path = st.text_input("Model Path", r"models\fer_model.pth")
data_dir = st.text_input("Data Save Directory", r"data\pictures")
device_option = st.radio("Select Device", ["GPU (0)", "CPU (-1)"])
device = torch.device("cuda:0" if "GPU" in device_option and torch.cuda.is_available() else "cpu")
adapt_training = st.checkbox("Enable Adaptive Training", value=False)

# Define the CNN model (must match trained model)
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
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

CLASS_NAMES = ['Boredom', 'Confused', 'Engaged', 'Frustration']

# Preprocessing function
def preprocess(face_img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    face_img = transform(face_img).unsqueeze(0)
    return face_img

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Create data directories
for emotion in CLASS_NAMES:
    os.makedirs(os.path.join(data_dir, emotion), exist_ok=True)

# Temporal smoothing for emotion predictions
emotion_history = deque(maxlen=5)  # Store last 5 predictions
last_emotion = None
transition_count = 0

# Adaptive training setup
if adapt_training:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    st.markdown("🧠 **Adaptive Training Enabled**: Approve frames to fine-tune the model.")

# Streamlit UI elements
start = st.button("🎥 Start Camera + Detect")
frame_window = st.image([])
result_placeholder = st.empty()
dimension_placeholder = st.empty()
adapt_frame_placeholder = st.empty()
save_frame_button = st.empty()

if start:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Failed to open camera. Please check your webcam.")
        st.stop()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("❌ Failed to load Haar cascade classifier.")
        st.stop()

    st.markdown("✅ **Camera Started. Detecting emotions and 3D face mesh...**")
    session_state = st.session_state
    if 'stop' not in session_state:
        session_state.stop = False

    stop_button = st.button("🛑 Stop Camera")
    if stop_button:
        session_state.stop = True

    while not session_state.stop:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Process 3D face mesh
        mesh_results = face_mesh.process(rgb_frame)
        face_dimensions = None
        if mesh_results.multi_face_landmarks:
            for landmarks in mesh_results.multi_face_landmarks:
                h, w = frame.shape[:2]
                # Calculate face dimensions (example: nose to chin distance)
                nose_tip = landmarks.landmark[4]  # Nose tip (MediaPipe landmark index 4)
                chin = landmarks.landmark[152]   # Chin (MediaPipe landmark index 152)
                nose_to_chin_px = abs(nose_tip.y * h - chin.y * h)
                face_width_px = abs(landmarks.landmark[234].x * w - landmarks.landmark[454].x * w)  # Left to right cheek
                face_dimensions = f"Face Width: {face_width_px:.1f}px, Nose-to-Chin: {nose_to_chin_px:.1f}px"
                # mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACE_MESH_TESSELATION, drawing_spec)
                mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=drawing_spec
                            )

        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            input_tensor = preprocess(face_img).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                predicted_class = CLASS_NAMES[torch.argmax(output, dim=1).item()]
            
            # Temporal smoothing
            emotion_history.append(predicted_class)
            emotion_counts = {cls: emotion_history.count(cls) for cls in CLASS_NAMES}
            smoothed_class = max(emotion_counts, key=emotion_counts.get)
            if smoothed_class != last_emotion and last_emotion is not None:
                transition_count += 1
                print(f"Emotion Transition #{transition_count}: {last_emotion} -> {smoothed_class}")
            last_emotion = smoothed_class

            # Draw face rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 50, 50), 2)
            cv2.putText(frame, smoothed_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display results
            result_placeholder.markdown(f"### 🧠 Detected Emotion: `{smoothed_class}` (Confidence: {probs[CLASS_NAMES.index(smoothed_class)]:.2%})")
            dimension_placeholder.markdown(f"### 📏 Face Dimensions: {face_dimensions}")

            # Save frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            frame_path = os.path.join(data_dir, smoothed_class, f"{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)

            # Adaptive training
            import uuid
            unique_key = f"train_{timestamp}_{uuid.uuid4().hex}"

            if adapt_training and save_frame_button.button(f"Use Frame for {smoothed_class} Training", key=unique_key):
                label = CLASS_NAMES.index(smoothed_class)
                input_tensor = input_tensor.to(device)
                label_tensor = torch.tensor([label], dtype=torch.long).to(device)
                optimizer.zero_grad()
                output = model(input_tensor)
                loss = criterion(output, label_tensor)
                loss.backward()
                optimizer.step()
                adapt_frame_placeholder.markdown(f"✅ Frame used to fine-tune model for `{smoothed_class}`.")

        frame_window.image(rgb_frame)

    cap.release()
    cv2.destroyAllWindows()
    st.markdown("🛑 **Camera Stopped.**")