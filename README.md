
# 🌿 ResTomato: ResNet50-Based Tomato Disease Classifier

ResTomato is a deep learning-based image classification project that uses a fine-tuned **ResNet50** architecture to detect and classify **tomato leaf diseases** from images. This system helps in early diagnosis of crop diseases, enabling timely intervention for better agricultural productivity and crop health.

---

## 📌 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Explainability](#-explainability)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## ✅ Features

- ⚙️ Transfer learning using **ResNet50** for high-accuracy classification.
- 📸 Classifies tomato leaf images into multiple **disease categories**.
- 📈 Generates confusion matrix, training/validation accuracy & loss plots.
- 📊 Includes **Exploratory Data Analysis (EDA)** for understanding dataset.
- 🧪 Adds **image augmentation** for better generalization.
- 🧠 Integrated with **LIME** for model explainability.
- 🖼️ Supports custom image prediction through command line.

---

## 📁 Project Structure

```
ResTomato/
│
├── Dataload.py          # Dataset loading and organization
├── EDA.py               # Dataset visualization & class balance
├── Preprocess.py        # Data augmentation and preprocessing
├── ResNet50.py          # Training the ResNet50 model
├── MatrixResNet50.py    # Evaluation via confusion matrix
├── Piechart.py          # Class distribution pie chart
├── plot.py              # Training/validation loss & accuracy plot
├── Predict.py           # Custom image prediction script
├── LIME.py              # Explainability using LIME
└── README.md            # Project documentation
```

---

## 🧠 Dataset

The dataset contains images of tomato leaves, categorized into several disease classes:

- Tomato___Bacterial_spot  
- Tomato___Early_blight  
- Tomato___Late_blight  
- Tomato___Leaf_Mold  
- Tomato___Septoria_leaf_spot  
- Tomato___Spider_mites  
- Tomato___Target_Spot  
- Tomato___Yellow_Leaf_Curl_Virus  
- Tomato___Mosaic_virus  
- Tomato___Healthy  

Ensure your dataset is organized as follows:

```
Dataset/
├── train/
│   ├── ClassA/
│   ├── ClassB/
│   └── ...
└── valid/
    ├── ClassA/
    ├── ClassB/
    └── ...
```
Link to the Dataset : 
```
https://www.kaggle.com/datasets/ashishmotwani/tomato
```

---

## ⚙️ Installation

### 1. Clone the Repository

```
git clone https://github.com/Himanshu49Gaur/ResTomato-ResNet50-Based-Tomato-Disease-Classifier.git
cd ResTomato-ResNet50-Based-Tomato-Disease-Classifier
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install packages manually:
```


---

## 🚀 Usage

### 1. Run Preprocessing and Training

```bash
python Dataload.py
python EDA.py
python Preprocess.py
python ResNet50.py
```

### 2. Evaluate the Model

```
python MatrixResNet50.py
python plot.py
python Piechart.py
```

### 3. Make Predictions on New Images

```
python Predict.py --image path/to/image.jpg
```

### 4. Explain Predictions with LIME

```
python LIME.py --image path/to/image.jpg
```

---

## 🏗️ Model Architecture

The model is based on **ResNet50**, a powerful pre-trained convolutional neural network from ImageNet. The top layer is replaced with:

- GlobalAveragePooling
- Dense layer with ReLU activation
- Output layer with Softmax for classification

> Fine-tuned with data augmentation and early stopping to avoid overfitting.

---

## 📊 Results

- **Accuracy**: Achieved >90% accuracy on validation set.
- **Confusion Matrix**: Highlights performance across all disease classes.
- **Training Curves**: Visualizations for loss and accuracy.
- **Model Robustness**: Generalizes well due to data augmentation.

---

## 🧠 Explainability

The model’s predictions can be interpreted using **LIME**, which generates a heatmap of the most influential pixels in the prediction decision.

- Helps build trust in the AI system.
- Useful for debugging model errors.

---

## 🧰 Technologies Used

| Tool/Library      | Purpose                             |
|-------------------|-------------------------------------|
| Python            | Core language                       |
| TensorFlow/Keras  | Model development                   |
| OpenCV            | Image processing                    |
| NumPy/Pandas      | Data handling                       |
| Matplotlib/Seaborn| Visualization                       |
| Scikit-learn      | Evaluation metrics                  |
| LIME              | Model interpretability              |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Himanshu Gaur**  
[GitHub Profile](https://github.com/Himanshu49Gaur)

---

## 🌱 Future Improvements

- Deploy the model using Streamlit or Flask.
- Create a mobile/web-friendly prediction interface.
- Expand the dataset to other crop species.
- Add real-time camera feed classification.

---
