"""
MLP Challenge Tester
--------------------
This script loads all submission files in the same directory and evaluates them
against a set of MNIST test images.

## **How It Works:**
1. Loads each Python file dynamically.
2. Extracts the participant's name using `get_participant_name()`.
3. Runs a set of test images using `predict(image)`.
4. Computes accuracy for each submission.
5. Outputs a CSV string containing participant names and scores.
"""

import os
import importlib.util
import torch
import torchvision
import torchvision.transforms as transforms

# ------------------------------------------------------
# 📌 Step 1: Load MNIST Test Data
# ------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# ------------------------------------------------------
# 📌 Step 2: Load All Submissions
# ------------------------------------------------------

submission_dir = os.path.dirname(os.path.abspath(__file__))  # Same directory as the tester
submission_files = [f for f in os.listdir(submission_dir) if f.endswith(".py") and f != "week1_challenge_tester.py"]

# Store results
results = []

for file in submission_files:
    # Load the submission module dynamically
    module_name = file[:-3]  # Remove .py extension
    file_path = os.path.join(submission_dir, file)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    submission_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submission_module)

    # Ensure the submission class is correctly implemented
    if not hasattr(submission_module, "MNISTClassifier"):
        print(f"❌ Skipping {file}: No class 'MNISTClassifier' found!")
        continue

    # Instantiate the class
    model_instance = submission_module.MNISTClassifier()

    # Check if required methods exist
    if not hasattr(model_instance, "get_participant_name") or not hasattr(model_instance, "predict"):
        print(f"❌ Skipping {file}: Missing required methods!")
        continue

    participant_name = model_instance.get_participant_name()

    # ------------------------------------------------------
    # 📌 Step 3: Evaluate Each Model
    # ------------------------------------------------------

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            for i in range(len(images)):
                image = images[i]
                label = labels[i].item()

                # Run prediction
                prediction = model_instance.predict(image)

                # Compare with actual label
                if prediction == label:
                    correct += 1

                total += 1

    # Compute Accuracy
    accuracy = correct / total * 100
    results.append((participant_name, accuracy))

# ------------------------------------------------------
# 📌 Step 4: Generate CSV Output
# ------------------------------------------------------

csv_output = "Participant,Accuracy\\n" + "\\n".join([f"{name},{score:.2f}" for name, score in results])

# Print final results
print("\\nFinal Results:")
print(csv_output)
