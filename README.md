🌄 Landscape Image Colorization using PyTorch
This project uses a Convolutional Neural Network (CNN) to automatically colorize grayscale landscape images. It is built using PyTorch and processes paired datasets of grayscale and color images to learn how to recreate color from black-and-white photos.
📁 Dataset Structure
css
Copy
Edit
landscape Images/
├── color/
│   ├── image1.jpg
│   └── ...
└── gray/
    ├── image1.jpg
    └── ...
📦 Features
Custom PyTorch Dataset (LandscapeDataset)

Grayscale and color image pairs

Preprocessing using torchvision.transforms

DataLoader for training and testing

Visualization function to compare gray vs. color

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
🧠 Model Architecture
The model is a Convolutional Neural Network (CNN) designed to learn the mapping from grayscale images to their colored counterparts. It consists of several convolutional layers followed by activation functions and upsampling layers to reconstruct the color image.​

🏋️ Training
To train the model, run the training script:​


python train.py
The training parameters are set as follows:​

Epochs: 3

Learning Rate: 0.001

Batch Size: 32

Device: CUDA if available, else CPU​

📊 Visualization
After training, you can visualize the results using the provided visualization script:​


python visualize.py
This will display a set of grayscale images alongside their colorized versions.



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


 











