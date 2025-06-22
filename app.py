import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class DigitGenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

@st.cache_resource
def load_model():
    model = DigitGenCNN()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
st.title("Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
if st.button("Generate Images"):
    fig, ax = plt.subplots(1, 5)
    for i in range(5):
        img = torch.randn(1, 1, 28, 28)
        pred = model(img).argmax().item()
        while pred != digit:
            img = torch.randn(1, 1, 28, 28)
            pred = model(img).argmax().item()
        ax[i].imshow(img.squeeze().detach().numpy(), cmap="gray")
        ax[i].set_title(f"Sample {i+1}")
        ax[i].axis("off")
    st.pyplot(fig)
