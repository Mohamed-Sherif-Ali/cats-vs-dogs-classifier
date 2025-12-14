# 🐕🐈 Dog vs Cat Image Classifier

**Deep Learning Binary Classification using Transfer Learning**

A high-accuracy image classifier using transfer learning with VGG16 and ResNet50 on the Cats vs Dogs dataset. Achieves **96%+ validation accuracy** with dual-branch ensemble architecture.

---

## ✨ Features

### 🎯 **Two Implementations**
- **VGG16 Baseline**: Single model approach (~93% accuracy)
- **Dual Ensemble**: VGG16 + ResNet50 combined (~96% accuracy)

### 🔥 **Key Capabilities**
- Transfer learning from ImageNet weights
- Data augmentation (horizontal flip + brightness)
- Real-time training visualization
- Confusion matrix analysis
- Sample prediction showcase
- Classification metrics (precision, recall, F1-score)

---

## 📊 Performance

| Model | Validation Accuracy | Training Time (GPU) | Parameters |
|-------|---------------------|---------------------|------------|
| **VGG16 Baseline** | ~93% | ~5-6 minutes | 14.7M |
| **Dual Ensemble** | **~96%** | ~8-10 minutes | 35.2M |

**Dataset**: 23,262 images (80% train / 20% validation)

---

## 🛠️ Technology Stack

- **Framework**: TensorFlow 2.13+ / Keras
- **Pre-trained Models**: VGG16, ResNet50 (ImageNet weights)
- **Dataset**: TensorFlow Datasets (Cats vs Dogs)
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: Scikit-learn

---

## 📦 Installation & Setup

### **Option 1: Google Colab (Recommended)** ⭐

**Advantages:**
- ✅ Free GPU (T4 with 16GB VRAM)
- ✅ All libraries pre-installed
- ✅ No setup required
- ✅ Run directly in browser

**Steps:**
1. Upload notebooks to Google Drive
2. Right-click notebook → Open with → Google Colaboratory
3. Runtime → Change runtime type → T4 GPU
4. Run all cells!

### **Option 2: Local Installation**

**Requirements:**
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU with 4GB+ VRAM (optional but recommended)

**Setup:**
```bash
# Clone or download repository
cd dog-cat-classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## 🚀 Quick Start

### **Step 1: Choose Your Model**

**For Beginners - Start with VGG16:**
```bash
jupyter notebook ANN_VGG16.ipynb
```

**For Best Accuracy - Use Dual Model:**
```bash
jupyter notebook ANN_dual.ipynb
```

### **Step 2: Run the Notebook**

1. Open the notebook in Jupyter/Colab
2. **Run all cells** (Cell → Run All) or press **Shift + Enter** for each cell
3. Wait for training to complete
4. View results and visualizations

### **Step 3: Results**

The notebook automatically generates:
- Training/validation accuracy curves
- Confusion matrix
- Sample predictions (correct vs incorrect)
- Classification report

---

## 📓 Notebook Overview

### **ANN_VGG16.ipynb** - VGG16 Baseline

**Architecture:**
```
Input (224×224×3)
    ↓
VGG16 Base (frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense (256, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (1, Sigmoid)
```

**Training Configuration:**
- Batch Size: 32
- Epochs: 10
- Optimizer: Adam
- Loss: Binary Crossentropy
- Data Augmentation: Horizontal flip + brightness

### **ANN_dual.ipynb** - Dual Ensemble Model

**Architecture:**
```
                Input (224×224×3)
                       ↓
         ┌─────────────┴─────────────┐
         ↓                           ↓
    VGG16 (frozen)              ResNet50 (frozen)
         ↓                           ↓
  GlobalAvgPool               GlobalAvgPool
         ↓                           ↓
     512 features                2048 features
         └─────────────┬─────────────┘
                       ↓
                 Concatenate (2560)
                       ↓
                Dense (256, ReLU)
                       ↓
                  Dropout (0.5)
                       ↓
                Dense (1, Sigmoid)
```

**Why Ensemble?**
- VGG16: Excellent at texture patterns
- ResNet50: Deep residual learning for complex features
- Combined: Complementary feature extraction → Better accuracy

---

## 📊 Dataset Information

### **Cats vs Dogs Dataset**
- **Source**: Microsoft Research
- **Total Images**: 23,262
- **Classes**: 2 (Cat = 0, Dog = 1)
- **Split**: 80% train (18,610) / 20% validation (4,652)
- **Original Sizes**: Variable (resized to 224×224)

### **Data Pipeline**
1. **Resize**: All images → 224×224 pixels
2. **Normalize**: Pixel values → [0, 1] range
3. **Augmentation** (training only):
   - Random horizontal flip
   - Random brightness adjustment (±10%)
4. **Batching**: 32 images per batch
5. **Prefetch**: Optimized loading

---

## 🎯 Training Process

### **Hyperparameters**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 32 | Balance speed/memory |
| Epochs | 10 | Sufficient for convergence |
| Learning Rate | 0.001 (Adam default) | Standard for fine-tuning |
| Image Size | 224×224 | VGG16/ResNet50 input size |
| Optimizer | Adam | Adaptive learning rate |
| Loss Function | Binary Crossentropy | Binary classification |

### **Training Steps**

1. **Load Dataset**: TensorFlow Datasets downloads automatically
2. **Preprocess**: Resize and normalize images
3. **Augment**: Apply random transformations to training data
4. **Build Model**: Load pre-trained weights, freeze base, add classifier
5. **Compile**: Set optimizer and loss function
6. **Train**: Fit model on training data
7. **Evaluate**: Test on validation set
8. **Visualize**: Plot results and predictions

---

## 📈 Results & Evaluation

### **Metrics Tracked**
- **Accuracy**: Correctly classified / Total
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

### **Visualizations Generated**

**1. Training Curves**
- Training vs Validation Accuracy
- Training vs Validation Loss
- Shows model convergence and potential overfitting

**2. Confusion Matrix**
```
              Predicted
              Cat    Dog
Actual Cat   [TN]   [FP]
      Dog    [FN]   [TP]
```

**3. Prediction Grid**
- 15 random sample images
- True label vs Predicted label
- Green = Correct, Red = Incorrect
- Confidence scores displayed

---

## 🔧 Customization & Tuning

### **Adjust Training Parameters**

```python
# In notebook, modify these variables:
BATCH_SIZE = 32        # Reduce if memory issues (try 16 or 8)
EPOCHS = 10            # Increase for better convergence (try 15-20)
```

### **Change Image Size**

```python
# Smaller images = faster training but lower accuracy
IMAGE_SIZE = (224, 224)  # Try (128, 128) for faster experiments
```

### **Modify Architecture**

```python
# VGG16 notebook - adjust dense layer size
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),  # Increase from 256
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

---

## 🐛 Troubleshooting

### **Issue: Out of Memory (OOM)**

**Error:** `ResourceExhaustedError: OOM when allocating tensor`

**Solutions:**
```python
# 1. Reduce batch size
BATCH_SIZE = 16  # or even 8

# 2. Use smaller images
IMAGE_SIZE = (128, 128)

# 3. For dual model, try VGG16 baseline first
```

### **Issue: Dataset Download Fails**

**Error:** `tfds.download.DownloadError`

**Solutions:**
- Check internet connection
- Retry download (sometimes servers are slow)
- Manual download:
  ```python
  tfds.load('cats_vs_dogs', download=True, data_dir='~/tensorflow_datasets')
  ```

### **Issue: Training Very Slow**

**Problem:** Training takes 60+ minutes

**Solutions:**
- **Use GPU**: In Colab, Runtime → Change runtime type → T4 GPU
- **Reduce epochs**: Start with 5 epochs for testing
- **Check GPU usage**:
  ```python
  import tensorflow as tf
  print("GPUs:", tf.config.list_physical_devices('GPU'))
  ```

### **Issue: Model Not Improving**

**Symptoms:** Accuracy stuck at ~50% (random guessing)

**Checks:**
```python
# 1. Verify labels
for images, labels in train_batches.take(1):
    print("Labels:", labels.numpy()[:10])
    # Should be mix of 0s and 1s

# 2. Check preprocessing
for images, _ in train_batches.take(1):
    print("Pixel range:", images.numpy().min(), "-", images.numpy().max())
    # Should be 0.0 - 1.0
```

---

## 📁 Project Structure

```
dog-cat-classifier/
│
├── ANN_VGG16.ipynb              # VGG16 baseline notebook
├── ANN_dual.ipynb               # Dual ensemble notebook
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
└── After running notebooks:
    ├── models/
    │   ├── cats_vs_dogs_vgg16_model.h5
    │   └── dual_final_*.h5
    │
    └── plots/
        ├── vgg16_confusion_matrix.png
        ├── vgg16_prediction_grid.png
        ├── dual_confusion_matrix.png
        └── dual_prediction_grid.png
```

---

## 🎓 Learning Outcomes

This project demonstrates:

- **Transfer Learning**: Leveraging pre-trained models
- **Ensemble Methods**: Combining multiple models
- **Data Augmentation**: Improving generalization
- **Model Evaluation**: Comprehensive metrics
- **Binary Classification**: Cat vs Dog prediction
- **Deep Learning Pipeline**: End-to-end workflow

---

## 🚀 Future Enhancements

Potential improvements:

- [ ] Fine-tuning: Unfreeze top layers of base models
- [ ] More architectures: EfficientNet, MobileNet
- [ ] K-fold cross-validation
- [ ] Test-time augmentation
- [ ] Grad-CAM visualization (see what model focuses on)
- [ ] Model deployment (Flask/FastAPI API)
- [ ] Web interface for predictions
- [ ] Mobile app with TensorFlow Lite

---

## 📚 References

**Models:**
- [VGG16 Paper](https://arxiv.org/abs/1409.1556) - Simonyan & Zisserman, 2014
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - He et al., 2015

**Dataset:**
- [Cats vs Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765) - Microsoft Research

**Framework:**
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)

---

## 💼 For Your Resume

**Suggested Description:**

```
Dog vs Cat Image Classifier (Deep Learning)
• Binary classification achieving 96%+ accuracy using transfer learning
• Dual-branch ensemble combining VGG16 + ResNet50 feature extraction
• Implemented data augmentation and preprocessing pipeline
• Comprehensive evaluation with confusion matrices and classification metrics
• Tech: TensorFlow, Keras, VGG16, ResNet50, Python, Jupyter
```


---

## 📄 License

MIT License - Feel free to use for learning and portfolio purposes.

---

## 👤 Author

**Mohamed Sherif Ali**  
 AI & Computer Vision Engineer  

---

## 🙏 Acknowledgments

- **TensorFlow Team** - Framework and pre-trained models
- **ImageNet** - Pre-trained weights
- **Microsoft Research** - Cats vs Dogs dataset
- **Google Colab** - Free GPU resources

---

## 📞 Support

**Questions or Issues?**
- Check the Troubleshooting section above
- Review TensorFlow documentation
- Test with smaller dataset first (use `train[:1%]` for quick tests)

---

**🐕 Happy Classifying! 🐈**

*Built with passion for deep learning and computer vision*
