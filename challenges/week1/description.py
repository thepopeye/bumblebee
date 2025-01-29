"""
Week 1 Challenge: MLP for MNIST Classification
----------------------------------------------
This challenge requires each participant to implement an MLP (Multi-Layer Perceptron) model using PyTorch.

## **Instructions:**
1. Implement a Python class that meets the following interface:
   - **get_participant_name()**: Returns the participant's name as a string.
   - **predict(image: torch.Tensor) -> int**: Accepts a single MNIST image (28x28) and returns a predicted digit (0-9).

2. Save your submission as a Python file in the `submissions/` directory. Name the file using your name/alias to keep it
unique.

3. The provided **tester** will automatically execute tests on your submission.
"""

# ------------------------------------------------------
# 📌 Step 1: Load and Preprocess MNIST Dataset
# ------------------------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define transformation: Convert to Tensor & Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

"""
### **Example Submission Format**
Participants must implement their solution in a class that follows this format:
"""


class MNISTClassifier:
    def __init__(self):
        """
        Initialize the model and load trained weights (if applicable).
        Participants should define their own model architecture.
        """
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        # Uncomment if loading trained weights
        # self.model.load_state_dict(torch.load("my_model.pth"))

    def get_participant_name(self):
        """ Returns the participant's name. """
        return "John Doe"  # Replace with your actual name

    def predict(self, image: torch.Tensor) -> int:
        """
        Accepts a single MNIST image, runs inference, and returns the predicted digit.
        """
        with torch.no_grad():
            output = self.model(image.view(1, 28 * 28))
            return output.argmax().item()


"""
## **Scoring Criteria**
- **Correct Predictions:** Each correct classification contributes to the final score.
- **Speed & Efficiency:** Submissions should execute efficiently.
- **Code Readability:** Clean, structured code is encouraged.

🚀 **Best of luck!** 🚀
"""
