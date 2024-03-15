import torch
import torchvision.transforms as transforms
from PIL import Image
from imagerecogtrain import Net

# Define the transform for the input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to match CIFAR-10 input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the saved state dictionary
#state_dict = torch.load('./cifar_net.pth', map_location=torch.device('cpu'))
state_dict = torch.load('./cifar_net.pth')
# Instantiate your model (assuming you're using the same architecture as before)
model = Net()
# Load the state dictionary into the model
model.load_state_dict(state_dict)
# Set the model to evaluation mode
model.eval()

# Define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Function to predict the class of an input image
def predict_image(image_path, model, transform):
    # Open and preprocess the input image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Get the predicted class label
    class_index = int(predicted.item())
    class_name = classes[class_index]
    return class_name

if __name__ == "__main__":
    # Path to the input image
    image_path = './bird.jpg'  # Change this to the path of your image

    # Predict the class of the input image
    predicted_class = predict_image(image_path, model, transform)
    print("Predicted class:", predicted_class)
