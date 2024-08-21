import torch
from torchvision import transforms
from PIL import Image

# Load the saved model
model = torch.load(r"C:\Users\FreeComp\Desktop\graduation project\DR code and data\Retino_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    # Load and preprocess the input image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Print the predicted class
    print("Predicted class:", predicted_class)

# Main function to test the prediction
def main():
    image_path = "dr.jpg"  # Update with the correct image path
    predict_image(image_path)

if __name__ == "__main__":
    main()
