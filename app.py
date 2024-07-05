import streamlit as st
import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
import gdown
import cv2
import os

# Download the model weights
url1 = 'https://drive.google.com/file/d/1yOPZyA14pBrInbM2jS63FX2J8lYpy0Sx/view?usp=drive_link'
output1 = 'ssd_model_weights.pth'

gdown.download(url1, output1, quiet=False)

# Function to load the model
def load_model(weights_path):
    # Initialize the model
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    # Load the weights into the model
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model

# Load the model
with st.spinner('Loading Model Into Memory....'):
    try:
        model = load_model(output1)
        st.success("Model Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

st.title("Weed Detection")
st.text("Upload a Plant Image for Weed Detection")

# File uploader for the image
img_file_buffer = st.file_uploader("Upload a Plant image....", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    # Read image file buffer with OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    def prediction(cv2_img):
        names = {0: 'crop', 1: 'weed'}
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()

        out = model(torch.unsqueeze(img_tensor, dim=0))
        boxes = out[0]['boxes'].cpu().detach().numpy().astype(int)
        labels = out[0]['labels'].cpu().detach().numpy()
        scores = out[0]['scores'].cpu().detach().numpy()
        Names = []

        for idx in range(boxes.shape[0]):
            if scores[idx] >= 0.8:
                x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
                name = names.get(labels[idx].item(), 'unknown')
                cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
                cv2.putText(cv2_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))
                Names.append(name)

        if "weed" in Names:
            st.write("Weed detected!")
        else:
            st.write("No weed detected.")

        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption='Processed Image.', use_column_width=True)

    prediction(cv2_img)

st.text("This project is developed by Ankesh Ansh")
