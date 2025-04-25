from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from model_configuration import ImprovedModel
import pickle
import os


with open("labels.pkl", "rb") as f:
    class_to_index = pickle.load(f)
    print(f"Number of classes: {len(set(class_to_index.values()))}")

# Tell Flask to look for templates in the same folder
app = Flask(__name__)

# Derive number of classes from label values
num_classes = len(set(class_to_index.values()))
print(f"Detected {num_classes} classes")

# Instantiate model with correct number of output classes
model = ImprovedModel(num_classes=num_classes)
# Load model
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

idx_to_class = {v: k for k, v in class_to_index.items()}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
        print(pred)

    print(pred.item())
    class_name_file_name = idx_to_class[pred.item()]
    
    # Path to matched images folder
    matched_path = os.path.join('static', 'matched', class_name_file_name)

    # Split by "\"
    path_split = matched_path.split("\\")
    print(path_split)
    
    # Get list of files in that class folder
    try:
        files = [f for f in path_split if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_name =  path_split[-2]
        filename = files[0] if files else None
    except FileNotFoundError:
        filename = None

    if filename:
        print("Returning:", {'class': class_name, 'filename': filename})
        return jsonify({'class': class_name, 'filename': filename})
    else:
        return jsonify({'error': 'No matching image found'})

if __name__ == '__main__':
    app.run(debug=True)
