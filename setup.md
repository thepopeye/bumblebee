# **Setup Instructions for Bumblebee Course Participants**

Welcome to **Bumblebee**, an open-source course on **Deep Learning & LLMs**! 🚀 Follow these steps to set up your environment and get started.

---

## **1. Fork & Clone the Repository**
Since course updates will be pushed to the repo, you should **fork** the repository first.

### **Step 1: Fork the Repo**
1. Visit: **[Bumblebee Repo](https://github.com/thepopeye/bumblebee)**
2. Click the **"Fork"** button (top-right) to create a personal copy of the repo in your GitHub account.

### **Step 2: Clone Your Fork Locally**
Replace `<your-github-username>` with your actual GitHub username.

```bash
git clone https://github.com/<your-github-username>/bumblebee.git
cd bumblebee
```

### **Step 3: Set Up Upstream (for Course Updates)**
To pull the latest course updates:

```bash
git remote add upstream https://github.com/thepopeye/bumblebee.git
git fetch upstream
git merge upstream/main
```

---

## **2. Install Dependencies**
We will use **PyTorch**, **Hugging Face Transformers**, and other tools.

### **Step 1: Create a Virtual Environment**
It’s recommended to use a virtual environment.

```bash
python -m venv bumblebee_env
source bumblebee_env/bin/activate  # On Windows: bumblebee_env\Scripts\activate
```

### **Step 2: Install Required Libraries**
Install dependencies using `pip`.

```bash
pip install torch torchvision torchaudio  # PyTorch
pip install numpy pandas matplotlib  # Core libraries
pip install transformers datasets  # Hugging Face tools
pip install scikit-learn  # Machine learning utilities
pip install tqdm  # Progress bars for training loops
```

---

## **3. Install & Configure PyCharm (Recommended IDE)**
We recommend **PyCharm** for development.

### **Step 1: Download PyCharm**
- Install **[PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/)** *(free version)*.

### **Step 2: Open the Repo in PyCharm**
1. Open **PyCharm** → Click **"Open Project"**.
2. Select the **`bumblebee`** folder.
3. Configure the **Python interpreter**:
   - Go to **File → Settings → Project → Python Interpreter**.
   - Select the virtual environment (`bumblebee_env`).

---

## **4. Running Tests**
Verify your setup by running a quick PyTorch test:

```python
import torch
print("CUDA Available:", torch.cuda.is_available())
```

If `CUDA Available: True`, your GPU is detected (for faster training). If `False`, PyTorch will use the CPU.

---

## **5. Keeping Your Fork Updated**
To pull the latest updates from the course repo:

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

---

### **✅ Next Steps**
Once set up:
1. **Join the discussions** (Zoom calls, issues, discussions in GitHub).
2. **Start working on the Week 1 guided project.**
3. **Track progress** by following the study plan.

---

🚀 **You're now ready to start the Bumblebee Deep Learning course!**

