import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import sys
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# Add src to python path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import Net

# --- Page Configuration ---
st.set_page_config(
    page_title="MNIST Neural Network",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Design ---
st.markdown("""
    <style>
    /* Dark Theme & Gradient Background */
    .stApp {
        background: rgb(15,23,42);
        background: linear-gradient(135deg, rgba(15,23,42,1) 0%, rgba(30,41,59,1) 100%);
        color: #e2e8f0;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #f8fafc !important;
        font-weight: 700;
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.5);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #0f172a;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    if not os.path.exists(model_path):
        return None
    
    device = torch.device("cpu")
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

# --- Main Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.title("🧠 Neural Network")
    st.markdown("### HandDigit Recognition")
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first.")
        st.stop()
    else:
        st.success("✨ System Online & Ready")

    st.markdown("---")
    mode = st.radio("Select Mode", ["🖌️ Draw", "🎲 Random Test"], index=0)

with col2:
    if mode == "🖌️ Draw":
        st.subheader("Draw a Digit (0-9)")
        
        # Canvas
        canvas_col, pred_col = st.columns([1, 1])
        
        with canvas_col:
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=20,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        
        with pred_col:
            if canvas_result.image_data is not None:
                # Preprocess
                img = canvas_result.image_data.astype('uint8')
                img = Image.fromarray(img)
                img = ImageOps.grayscale(img)
                img = img.resize((28, 28))
                
                # Convert to Tensor
                tensor = transforms.ToTensor()(img)
                tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
                
                # Predict
                # Check if image is not empty (black)
                # We check raw numpy array max value
                if np.array(img).max() > 10: 
                    with torch.no_grad():
                        output = model(tensor.unsqueeze(0))
                        pred = output.argmax(dim=1, keepdim=True).item()
                        confidence = torch.exp(output).max().item()
                    
                    st.markdown(f"""
                        <div style="text-align: center; background-color: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;">
                            <h3 style="color: #94a3b8; margin:0; font-size: 1rem;">Prediction</h3>
                            <h1 style="color: #ffffff; font-size: 4rem; margin:0; text-shadow: 0 0 20px rgba(56, 189, 248, 0.5);">{pred}</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence:.2%}")
                else:
                    st.info("Draw something to get a prediction!")

    elif mode == "🎲 Random Test":
        st.subheader("Random Test Samples")
        if st.button("Generate New Samples"):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
            
            indices = np.random.choice(len(dataset), 5, replace=False)
            cols = st.columns(5)
            
            for i, idx in enumerate(indices):
                img, label = dataset[idx]
                with torch.no_grad():
                    output = model(img.unsqueeze(0))
                    pred = output.argmax(dim=1, keepdim=True).item()
                
                with cols[i]:
                    img_disp = img.squeeze().numpy()
                    fig, ax = plt.subplots(figsize=(2,2))
                    ax.imshow(img_disp, cmap='gray')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    color = "#4ade80" if pred == label else "#f87171"
                    st.markdown(f"<div style='text-align:center; color:{color}; font-weight:bold; font-size:1.5em'>{pred}</div>", unsafe_allow_html=True)
                    st.caption(f"True: {label}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #94a3b8;'>Powered by PyTorch DDP & Kubeflow</div>", unsafe_allow_html=True)
