import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import pathlib
import logging
import time
import uuid
import plotly.express as px
import pandas as pd
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    filename="emotion_detection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Streamlit page configuration
st.set_page_config(
    page_title="Emotion Detection with 3D Face Mesh",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_path = st.text_input(
        "Model Path",
        value=r"D:\BINs\friends\FER\models\fer_model.pth",
        help="Path to PyTorch model (.pth)"
    )
    save_dir = st.text_input(
        "Save Directory",
        value=str(pathlib.Path.home() / "emotion_data"),
        help="Directory to save outputs"
    )
    device_option = st.radio("Device", ["CPU", "GPU (cuda)"], index=0)
    st.markdown("---")
    st.subheader("Options")
    show_landmarks = st.checkbox("Show Face Landmarks", value=True)
    save_output = st.checkbox("Save Outputs", value=False)
    st.markdown("---")
    st.subheader("About")
    st.markdown(
        """
        Upload an image to detect emotions and view a 3D face mesh.
        Emotions: **Boredom**, **Confused**, **Engaged**, **Frustration**.
        """
    )

# Validate inputs
model_path = pathlib.Path(model_path)
if not model_path.is_file():
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()
save_dir = pathlib.Path(save_dir)
if save_output and not save_dir.is_dir():
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created save directory: {save_dir}")
    except Exception as e:
        st.error(f"❌ Failed to create save directory: {e}")
        logger.error(f"Save directory creation failed: {e}")
        st.stop()

# Set device
device = torch.device("cuda" if device_option == "GPU (cuda)" and torch.cuda.is_available() else "cpu")
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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    st.success(f"✅ Model loaded: {model_path}")
    logger.info(f"Model loaded from {model_path}")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    logger.error(f"Model loading failed: {e}")
    st.stop()

CLASS_NAMES = ['Boredom', 'Confused', 'Engaged', 'Frustration']

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
try:
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    logger.info("MediaPipe Face Mesh initialized")
except Exception as e:
    st.error(f"❌ Failed to initialize MediaPipe: {e}")
    logger.error(f"MediaPipe initialization failed: {e}")
    st.stop()

# Preprocessing function
def preprocess(img):
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = np.array(img.convert("RGB"))
        img_tensor = transform(img).unsqueeze(0)
        if img_tensor.shape != (1, 3, 224, 224):
            raise ValueError("Invalid image shape")
        return img_tensor
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None

# 3D face plot function
@st.cache_data
def plot_3d_face(landmarks, frame_shape, _cache_key):
    try:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        h, w = frame_shape[:2]
        x = [l.x * w for l in landmarks.landmark]
        y = [l.y * h for l in landmarks.landmark]
        z = [l.z * w for l in landmarks.landmark]
        ax.scatter(x, y, z, s=2, c='b', label='Landmarks')
        for conn in mp_face_mesh.FACEMESH_TESSELATION:
            start, end = conn
            ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'g', linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Face Mesh')
        ax.view_init(elev=10, azim=45)
        ax.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        logger.error(f"3D plot error: {e}")
        return None

# Main UI
st.title("🎭 Emotion Detection with 3D Face Mesh")
st.markdown("Upload an image to predict emotions and view a 3D face mesh.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📷 Image & Prediction", "📊 Confidence", "🧠 3D Mesh"])

# Image uploader
with tab1:
    img_file = st.file_uploader(
        "📤 Upload Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear face image",
        key="image_uploader"
    )

# Process image
if img_file:
    try:
        image = Image.open(img_file).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_for_mesh = img_bgr.copy()

        # MediaPipe face mesh
        face_detected = False
        mesh_results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if show_landmarks and mesh_results.multi_face_landmarks:
            for landmarks in mesh_results.multi_face_landmarks:
                face_detected = True
                mp_drawing.draw_landmarks(
                    image=img_for_mesh,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec
                )
            img_for_mesh = cv2.cvtColor(img_for_mesh, cv2.COLOR_BGR2RGB)

        # Display image
        with tab1:
            st.image(
                img_for_mesh if show_landmarks and face_detected else image,
                caption="Processed Image" if face_detected else "Uploaded Image",
                use_column_width=True
            )

        # Emotion prediction
        img_tensor = preprocess(image)
        probs = None
        if img_tensor is not None:
            with tab1:
                with st.spinner("🧠 Predicting..."):
                    try:
                        img_tensor = img_tensor.to(device)
                        with torch.no_grad():
                            output = model(img_tensor)
                            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                            predicted_class = CLASS_NAMES[np.argmax(probs)]
                            confidence = np.max(probs) * 100
                        st.success(f"**Emotion**: {predicted_class}")
                        st.metric("Confidence", f"{confidence:.2f}%")
                        logger.info(f"Predicted {predicted_class} with confidence {confidence:.2f}%")
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                        logger.error(f"Prediction error: {e}")

        # Save processed image
        if save_output and face_detected:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_save_path = save_dir / f"processed_{timestamp}.jpg"
            try:
                cv2.imwrite(str(img_save_path), cv2.cvtColor(img_for_mesh, cv2.COLOR_RGB2BGR))
                st.info(f"📸 Saved image: {img_save_path}")
                logger.info(f"Saved image: {img_save_path}")
            except Exception as e:
                st.error(f"❌ Failed to save image: {e}")
                logger.error(f"Image save failed: {e}")

        # Confidence scores
        with tab2:
            if probs is not None:
                df = pd.DataFrame({"Emotion": CLASS_NAMES, "Confidence": probs * 100})
                fig = px.bar(
                    df,
                    x="Emotion",
                    y="Confidence",
                    color="Emotion",
                    title="Confidence Scores",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_layout(showlegend=False, yaxis_title="Confidence (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No prediction available.")

        # 3D face mesh
        with tab3:
            if mesh_results.multi_face_landmarks:
                for landmarks in mesh_results.multi_face_landmarks:
                    cache_key = str(uuid.uuid4())
                    fig_3d = plot_3d_face(landmarks, img_bgr.shape, _cache_key=cache_key)
                    if fig_3d:
                        st.image(fig_3d, caption="3D Face Mesh", use_column_width=True)
                        if save_output:
                            plot_save_path = save_dir / f"3d_plot_{timestamp}.png"
                            try:
                                fig_3d.save(str(plot_save_path))
                                st.info(f"📊 Saved 3D plot: {plot_save_path}")
                                logger.info(f"Saved 3D plot: {plot_save_path}")
                            except Exception as e:
                                st.error(f"❌ Failed to save 3D plot: {e}")
                                logger.error(f"3D plot save failed: {e}")
                    else:
                        st.error("❌ Failed to generate 3D mesh")
            else:
                st.warning("No face detected. Upload a clearer face image.")

    except Exception as e:
        st.error(f"❌ Image processing failed: {e}")
        logger.error(f"Image processing failed: {e}")
    finally:
        if 'image' in locals():
            image.close()

    # Clear button
    with tab1:
        if st.button("🗑️ Clear Image", key="clear_image"):
            st.session_state.pop("image_uploader", None)
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        <p>Built with ❤️ using Streamlit, PyTorch, and MediaPipe | <a href='https://x.ai'>xAI</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Cleanup
try:
    face_mesh.close()
    logger.info("MediaPipe Face Mesh closed")
except Exception as e:
    logger.error(f"MediaPipe cleanup failed: {e}")