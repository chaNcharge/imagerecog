import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Redefine same neural network model from the training script
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Increase number of channels
        self.bn1 = nn.BatchNorm2d(32)                # Add Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # Increase number of channels
        self.bn2 = nn.BatchNorm2d(64)                # Add Batch Normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # Increase number of channels
        self.bn3 = nn.BatchNorm2d(128)                # Add Batch Normalization
        self.fc1 = nn.Linear(128 * 4 * 4, 512)        # Increase number of units
        self.fc2 = nn.Linear(512, 256)                # Decrease number of units
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    class_index = predicted.item()
    class_name = classes[class_index]
    return class_name

if __name__ == "__main__":
    # Path to the input image
    image_path = './bird.jpg'  # Change this to the path of your image

    # Predict the class of the input image
    predicted_class = predict_image(image_path, model, transform)
    print("Predicted class:", predicted_class)
