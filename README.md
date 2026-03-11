# MNIST Digit Recognition Web App
## A CNN-powered handwritten digit recognizer with an interactive drawing canvas

This project is a simple end-to-end machine learning application demonstrating how a trained Convolutional Neural Network (CNN) can be deployed as a web service.

Users can draw a digit (0–9) directly in the browser using an interactive canvas, and the model predicts the digit in real time.

The backend is built with Flask and PyTorch, while the frontend uses HTML, CSS, and JavaScript with the browser Canvas API. The model was trained on the MNIST handwritten digit dataset and achieves around ~99% accuracy on the test set.

The project demonstrates the complete ML inference pipeline:

```
User Drawing
      ↓
Image Preprocessing
      ↓
Neural Network Inference
      ↓
Prediction Response
```

## Live Demo

The application is deployed online and can be accessed here:

https://mnist-cnn-digit-recognition-webapp.onrender.com/

Simply open the link and start drawing digits.

No installation required.

## Visual Demonstration
![App Screenshot](/screenshot/app-screenshot.png)

Example workflow:

Draw a number on the canvas

Click Predict

The model returns the predicted digit and confidence score

## Installation Instructions (For Users)

If you simply want to use the application, open the live deployment:
```https://mnist-cnn-digit-recognition-webapp.onrender.com/```
Draw a digit in the canvas and press Predict.

No setup is required.


## Installation Instructions (For Collaborators / Developers)

If you want to run the project locally and modify the code, follow these steps.

1. Clone the repository
```
git clone https://github.com/YOUR_USERNAME/mnist-digit-recognition-webapp.git
cd mnist-digit-recognition-webapp
```

2. Create a virtual environment
```
python -m venv venv
```

3. Activate the virtual environment
```
Windows

venv\Scripts\activate

Mac / Linux

source venv/bin/activate
```

4. Install dependencies
```
pip install -r requirements.txt
```
5. Run the Flask development server
```
python app.py
```
6. Open the application
```
Visit:

http://127.0.0.1:5000
```
You should now see the digit drawing canvas running locally.

## Expectations for Contributors

Contributions are welcome. If you would like to improve the project, please follow these guidelines.

### Code Standards

• Keep functions small and readable
• Add comments for complex or non-obvious logic
• Follow consistent naming conventions

### Suggested Areas for Improvement

Possible improvements include:

• Better digit preprocessing
• Improved UI / UX
• Support for uploaded images
• Model architecture improvements
• Mobile-friendly interface

### Contribution Workflow

1. Fork the repository

2. Create a feature branch
```git checkout -b feature-name```

3. Commit your changes

```git commit -m "Add feature description"```

4. Push the branch

```git push origin feature-name```

5. Open a Pull Request

## Known Issues
1. Bounding Box Centering (Not Center-of-Mass)

The current preprocessing centers digits using a bounding box approach, meaning the smallest rectangle containing all digit pixels is cropped and centered.

However, the original MNIST preprocessing uses center-of-mass centering, which can provide better alignment for skewed or asymmetrical digits.

Possible improvement:

• Implement center-of-mass based centering.

2. Rotation Sensitivity

The model was trained only on upright digits, so rotated or upside-down digits may be misclassified.

Possible improvement:

• Use data augmentation with rotations during training.

3. Stroke Thickness Variability

Canvas stroke thickness may vary between users, which can slightly affect prediction quality.

Possible improvements:

• Normalize stroke width
• Apply morphological preprocessing

4. Distribution Mismatch

The model expects MNIST-style inputs:
```
white digit on black background
centered
28 × 28 image
```

User drawings may deviate from this distribution, which can reduce prediction accuracy.

Improved preprocessing could reduce this mismatch.

Technologies Used

• Python
• PyTorch
• Flask
• HTML / CSS / JavaScript
• MNIST Dataset

License

This project is open-source and available under the MIT License.

Author

Abdul-Hameed Zubair

⚜ ÎnFî-KnÎght ⚜