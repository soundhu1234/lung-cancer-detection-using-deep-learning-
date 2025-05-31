import os
from flask import Flask, render_template, request
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

# Ensure the charts directory exists
CHARTS_DIR = "static/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained ViT model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=4,  # Model trained for 4 classes
    ignore_mismatched_sizes=True
)
model = model.to(device)

# Load the trained model weights
checkpoint = torch.load('lung_cancer_vit_model.pth', map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()  # Set model to evaluation mode

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Class names
class_names = [
    'Adenocarcinoma',
    'Large Cell Carcinoma',
    'Normal',
    'Squamous Cell Carcinoma'
]

# Function to predict image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    return predicted_class.item(), confidence.item()

# Function to create and save a bar chart
def save_prediction_chart(predicted_class, confidence):
    values = [0, 0, 0, 0]  # Initialize all values to zero
    values[predicted_class] = confidence * 100  # Set the detected class with confidence

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, values, color=['red', 'orange', 'green', 'blue'])
    plt.xlabel("Lung Cancer Types")
    plt.ylabel("Confidence Score (%)")
    plt.title("Lung Cancer Prediction")

    chart_path = os.path.join(CHARTS_DIR, "prediction_chart.png")
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        name = request.form.get('name')  # Get name from form
        age = request.form.get('age')  # Get age from form
        symptoms = request.form.getlist('symptoms')  # Get symptoms as a list

        if file and name and age:
            # Save uploaded image
            image_path = os.path.join('static', 'uploads', file.filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure directory exists
            file.save(image_path)

            # Predict class
            predicted_class, confidence = predict_image(image_path)
            predicted_label = class_names[predicted_class]

            # Generate prediction chart
            chart_path = save_prediction_chart(predicted_class, confidence)

            # Pass symptoms data to the result page
            return render_template(
                'result.html',
                name=name,
                age=age,
                predicted_class=predicted_label,
                confidence=round(confidence * 100, 2),
                image_path=image_path,
                chart_path=chart_path,
                symptoms=symptoms  # Sending symptoms to result.html
            )
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)