# 🐱🐶 Cats vs Dogs Classifier — Dual CNN (VGG16 + ResNet50)

A high-accuracy binary image classifier trained using transfer learning on the Cats vs Dogs dataset.  
Achieves 96%+ validation accuracy with a dual-branch VGG16 + ResNet50 architecture.

---

## 🚀 Features
- Dual CNN architecture combining two pre-trained backbones
- Fine-tuned with early stopping and data augmentation
- Visualization: accuracy/loss curves, confusion matrix
- Comparison with standalone VGG16 baseline

---

## 🧠 Tools & Libraries
- Python · TensorFlow · Keras · Matplotlib · Scikit-learn

---

## 📊 Performance
- **Validation Accuracy**: ~96.15%
- **Single VGG16 Accuracy**: ~93%
- Plots in `/plots/` folder

---

## ▶️ To Run
```bash
pip install -r requirements.txt
jupyter notebook model_dual_vgg16_resnet50.ipynb
