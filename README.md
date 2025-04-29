🌄 Landscape Image Colorization using PyTorch
This project uses a Convolutional Neural Network (CNN) to automatically colorize grayscale landscape images. It is built using PyTorch and processes paired datasets of grayscale and color images to learn how to recreate color from black-and-white photos.

📁 Dataset Structure
Make sure your dataset is structured like this:

css
Copy
Edit
landscape Images/
├── color/  → RGB colored images
└── gray/   → Grayscale images (same filenames as in color/)
 Requirements
Install these Python packages:

bash
Copy
Edit
pip install torch torchvision matplotlib pillow tqdm
If you're using Google Colab, mount your Google Drive and unzip the dataset:

python
Copy
Edit
from google.colab import drive
drive.mount('/content/gdrive')
!unzip "/content/gdrive/MyDrive/archive (3).zip" -d "/content/"
🧠 Model Description
The model is a basic CNN that takes 1-channel grayscale images as input and tries to generate 3-channel RGB color images. It is trained using pairs of grayscale and corresponding color images.

Optimizer: SGD

Loss Function: MSELoss

Epochs: 3

Batch Size: 32

Device: GPU/CPU (auto-detect)

🚀 How to Run
Load and prepare the dataset using the custom LandscapeDataset class.

Split the dataset into training and testing sets.

Train the CNN model using DataLoader.

Visualize grayscale vs color output using show_images() function.

💻 Optional: Deploy with Gradio
You can build a web app using Gradio to upload grayscale images and view the colorized result.

bash
Copy
Edit
pip install gradio
Then use a script like this:

python
Copy
Edit
import gradio as gr

def predict(img):
    # Preprocess and run model prediction
    return colorized_img

gr.Interface(fn=predict, inputs="image", outputs="image").launch()
📷 Output Sample
Use the visualization function to compare grayscale input and model-predicted color output side-by-side.

📬 Contact
Created by Megha Solanki
For questions or collaboration, connect via GitHub Issues











