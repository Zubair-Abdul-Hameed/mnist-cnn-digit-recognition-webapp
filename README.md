# MNIST Digit Recognition Web App
### A CNN-powered handwritten digit recognizer with an interactive drawing canvas

This project is a simple end-to-end machine learning application that demonstrates how a trained Convolutional Neural Network (CNN) can be deployed as a web service. Users can draw a digit (0–9) directly on a canvas in the browser, and the model predicts the digit in real time.

The backend is built using **Flask** and **PyTorch**, while the frontend uses **HTML, CSS, and JavaScript** with the browser Canvas API. The model was trained on the **MNIST handwritten digit dataset** and achieves around **~99% accuracy** on the test set.

The main goal of this project is to demonstrate the full ML pipeline:


User drawing → Image preprocessing → Neural network inference → Prediction response


---

# Visual Demonstration

![App Screenshot](IMAGE_LINK_HERE)

Example workflow:

1. Draw a number on the canvas
2. Click **Predict**
3. The model returns its predicted digit and confidence score

---

# Installation Instructions (For Users)

If you simply want to run the app locally.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mnist-digit-recognition-webapp.git
cd mnist-digit-recognition-webapp
2. Create a virtual environment
python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac / Linux

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Run the application
python app.py
5. Open the browser

Visit:

http://127.0.0.1:5000

You should now see the drawing canvas.

Installation Instructions (For Collaborators / Developers)

If you want to modify the model, frontend, or deployment logic.

Project Structure
mnist-digit-recognition-webapp/
│
├── app.py                 # Flask backend server
├── mnist_cnn.py           # Model training script
├── mnist_cnn.pth          # Trained model weights
├── requirements.txt
│
├── templates/
│   └── index.html         # Frontend interface
│
└── README.md
Development Setup

Clone the repository

git clone https://github.com/YOUR_USERNAME/mnist-digit-recognition-webapp.git

Create a virtual environment

python -m venv venv

Activate it

Install dependencies

pip install -r requirements.txt

Run the Flask development server

python app.py
Training the Model

If you want to retrain the model:

python mnist_cnn.py

This will generate a new mnist_cnn.pth file.

Expectations for Contributors

Contributions are welcome. If you'd like to improve the project, please follow these guidelines.

Code Standards

Keep functions small and readable

Add comments for non-obvious logic

Follow consistent naming conventions

Suggested Areas for Improvement

Some possible improvements include:

Better digit preprocessing

Improved UI/UX

Support for uploaded images

Model improvements

Mobile-friendly interface

Contribution Workflow

Fork the repository

Create a feature branch

git checkout -b feature-name

Commit your changes

git commit -m "Add feature description"

Push the branch

git push origin feature-name

Open a Pull Request

Known Issues
1. Bounding Box Centering (Not Center-of-Mass)

The current preprocessing centers the digit using a bounding box approach, meaning the smallest rectangle containing all digit pixels is cropped and centered.

However, the original MNIST preprocessing uses center-of-mass centering, which can produce slightly better alignment for digits that are skewed or asymmetrical.

Future improvement:

Implement center-of-mass based centering.

2. Rotation Sensitivity

The model was trained on upright digits only, so rotated or upside-down digits may be misclassified.

Possible solution:

Add data augmentation with rotations during training.

3. Stroke Thickness Variability

Canvas stroke thickness may vary between users, which can slightly affect prediction quality.

Possible improvements:

Normalize stroke width

Apply morphological preprocessing

4. Distribution Mismatch

The model expects MNIST-style inputs:

white digit on black background
centered
28×28 image

User drawings may deviate from this distribution, which can reduce accuracy.

Better preprocessing could reduce this issue.

Technologies Used

Python

PyTorch

Flask

HTML / CSS / JavaScript

MNIST Dataset

License

This project is open-source and available under the MIT License.
