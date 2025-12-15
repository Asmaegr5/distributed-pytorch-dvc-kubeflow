import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import sys
import os

# Add src to python path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import Net

st.title("MNIST Distributed Training Demo")
st.write("This app visualizes the model trained via PyTorch DDP on Kubeflow.")

model_path = "model.pth"

if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please train the model first.")
else:
    # Load Model
    device = torch.device("cpu")
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    st.success("Model loaded successfully!")

    # Load Data Samples
    st.subheader("Inference on Test Data")
    
    if st.button("Show Random Predictions"):
        # We process on the fly for demo
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download strictly to cache just for this demo if not present
        if not os.path.exists('data'):
             st.warning("Data directory not found. Downloading...")
        
        try:
            dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            
            indices = np.random.choice(len(dataset), 5, replace=False)
            
            cols = st.columns(5)
            
            for i, idx in enumerate(indices):
                img, label = dataset[idx]
                with torch.no_grad():
                    output = model(img.unsqueeze(0))
                    pred = output.argmax(dim=1, keepdim=True).item()
                
                with cols[i]:
                    # Unnormalize for display
                    img_disp = img.squeeze().numpy()
                    plt.figure(figsize=(2, 2))
                    plt.imshow(img_disp, cmap='gray')
                    plt.axis('off')
                    st.pyplot(plt)
                    
                    st.write(f"**Pred:** {pred}")
                    st.write(f"True: {label}")
                    if pred == label:
                        st.write("✅")
                    else:
                        st.write("❌")

        except Exception as e:
            st.error(f"Error loading data: {e}")

st.markdown("---")
st.markdown("### System Info")
st.write(f"PyTorch Version: {torch.__version__}")
