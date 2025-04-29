🌄 Landscape Image Colorization using PyTorch
This project uses a Convolutional Neural Network (CNN) to automatically colorize grayscale landscape images. It is built using PyTorch and processes paired datasets of grayscale and color images to learn how to recreate color from black-and-white photos.

🚀 Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/image-colorization.git
cd image-colorization

3. Mount Google Drive in Colab (if using Colab)
python
Copy
Edit
from google.colab import drive
drive.mount('/content/gdrive')

5. Extract the Dataset
bash
Copy
Edit
!unzip "/content/gdrive/MyDrive/archive (3).zip" -d "/content/"

7. Dataset Structure
Make sure your dataset is structured like this:

nginx
Copy
Edit
landscape Images/
├── color/  # colored images
└── gray/   # grayscale images (same filenames as in color/)

⚙️ Dependencies
Install these required libraries:

bash
Copy
Edit
pip install torch torchvision matplotlib tqdm pillow gradio

🧠 Project Workflow
Custom Dataset Loader using PyTorch

Preprocessing & Normalization

CNN Model to map grayscale → color

Training & Evaluation on custom images

Visualization of results (grayscale vs. predicted color)

Gradio Interface (optional) for web-based testing

🌐 Web Demo (Optional)
To run a browser-based demo using Gradio:

python
Copy
Edit
import gradio as gr

def predict(image):
    # Your model inference code here
    return colorized_image

gr.Interface(fn=predict, inputs="image", outputs="image").launch()
📸 Example Output

Grayscale Input	Colorized Output
📬 Contact
Created by Megha Solanki
For questions, open an issue on GitHub.


 











