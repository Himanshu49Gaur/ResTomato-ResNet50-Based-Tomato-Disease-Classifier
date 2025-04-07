# 🧠 How to Install CUDA and Link it with PyTorch for Model Training

This guide walks you through installing NVIDIA CUDA and configuring PyTorch to use it for accelerated model training on a GPU.

---

## 🚀 Step 1: Check Your GPU Compatibility

Before installing CUDA, ensure your system has an NVIDIA GPU:

```bash
nvidia-smi
```

If this command works and lists your GPU, you're good to proceed.

---

## 📥 Step 2: Install CUDA Toolkit

1. **Go to the official CUDA Toolkit download page:**
   👉 [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

2. **Select your OS**, architecture, distribution, and version.

3. **Follow the instructions provided** for your platform. For example, on Ubuntu:

```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

> ✅ Make sure the version you install is compatible with the PyTorch version you want to use.

---

## ✅ Step 3: Verify CUDA Installation

Run:

```bash
nvcc --version
```

You should see output showing the installed CUDA version (e.g., 12.1).

Also run:

```bash
nvidia-smi
```

This should show your GPU status and the compatible CUDA version.

---

## 🧪 Step 4: Install PyTorch with CUDA Support

Visit the official PyTorch Get Started page:
👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Choose the appropriate settings (OS, Package, Language, CUDA version) and it will generate an install command like:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with the version matching your CUDA installation.

---

## 🔍 Step 5: Verify PyTorch CUDA Integration

Open Python or a Jupyter Notebook and run:

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Shows your GPU name
```

---

## 🏋️ Step 6: Train a Model Using the GPU

Here's a simple example:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YourModel().to(device)
data = data.to(device)
target = target.to(device)

output = model(data)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
```

---

## 🛠️ Troubleshooting

- ❌ `torch.cuda.is_available()` returns `False`:
  - Check if the installed PyTorch version supports your CUDA version.
  - Verify your GPU driver is up to date.
  - Restart your system after CUDA installation.

---

## 🎉 You're All Set!

You can now train deep learning models using GPU acceleration with PyTorch and CUDA. Happy coding! 🧑‍💻⚡
