# Bridging the Domain Gap: Synthetic-to-Real Image Classification with Domain Adaptation

This repository contains a complete pipeline for training an image classifier on synthetic images and adapting it for real-world performance using advanced domain adaptation techniques.

The project explores how synthetic datasets—generated using prompt-based diffusion models—can be leveraged to train robust classifiers, with further enhancements through Conditional Domain Adversarial Networks (CDAN), pseudo-labeling, spectral normalization, and consistency regularization.

---

## 📌 File Structure
- image1000/synthetic_dataset # Pre-generated synthetic dataset (airplane, bird, car)
- stl10_png/ # STL10 real-world dataset (labeled + unlabeled)
- CODE.py 
- REPORT.pdf
- README.md # This documentation

---

## 🚀 Key Components

### ✅ Step 1: Training on Synthetic Images
- Fine-tuned a modified ResNet18 on images generated using Stable Diffusion.
- Heavy data augmentation with RandAugment, dropout, and cosine annealing learning rate.
- Saved best model based on synthetic validation accuracy.

### ✅ Step 2: Testing on Real Data
- STL10 real dataset used for testing.
- Test-time augmentation using FiveCrop to simulate robustness.
- Evaluated accuracy and performance gap between synthetic and real domains.

### ✅ Step 3: Domain Adaptation with CDAN
- Generated pseudo-labels on unlabeled real images with sharpened softmax.
- Used a domain discriminator trained adversarially with a Gradient Reversal Layer (GRL).
- Integrated consistency regularization using weak–strong augmentations.
- Spectral normalization for training stability.

---

## 📊 Results

| Dataset            | Accuracy (%) |
|--------------------|--------------|
| Synthetic Validation | ~99.8       |
| Real Test (pre-DA)   | ~93.5       |
| Real Test (CDAN + Consistency) | **95.6** |

Performance gap reduced by:
- Absolute: ~2.3%
- Relative: ~23% improvement over initial domain gap

---

## 📁 Datasets

- `image1000/`: 1000 synthetic images (3-class: airplane, bird, car)
- `stl10_png/`: Real STL10 dataset (split into labeled and unlabeled)

---

## 🧠 Insights

- Synthetic data alone achieves strong generalization, but still underperforms on real-world distributions.
- Adversarial adaptation and consistency regularization are critical for real-world robustness.
- Simple sharpening + thresholding can yield high-quality pseudo-labels even with OOD images.

---

### Run:

```bash
python CODE.PY
