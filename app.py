import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

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

@st.cache_resource
def load_data():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    return dataset

model = load_model()
data = load_data()

st.title("Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

if st.button("Generate Images"):
    fig, ax = plt.subplots(1, 5, figsize=(10, 2))
    count = 0
    for img, label in data:
        if label == digit:
            ax[count].imshow(img.squeeze(), cmap="gray")
            ax[count].set_title(f"Sample {count+1}")
            ax[count].axis("off")
            count += 1
        if count == 5:
            break
    st.pyplot(fig)
