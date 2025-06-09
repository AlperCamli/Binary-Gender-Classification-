# VGG-16 Transfer Learning for Binary Gender Classification

This project applies **transfer learning** using the **VGG-16** convolutional neural network to classify binary gender (Male/Female) based on facial images. The model is trained and evaluated on a subset of the **CelebA** dataset.

> **Key Focus:** Explore the effects of different learning rates and fine-tuning strategies on classification performance.

## Project Structure

- `notebooks/`: Jupyter notebooks with training and evaluation experiments
- `data/`: Preprocessed CelebA30k subset (not included due to licensing)
- `models/`: VGG-16 based configurations
- `results/`: Plots, confusion matrices, metrics
- `README.md`: Project overview
- `requirements.txt`: Python dependencies

## Dataset

A custom subset of **CelebA** dataset (`CelebA30k`) with **30,000 facial images**, each annotated with multiple binary attributes. This project focuses only on the "Male" column.

- Train: 24,000 images (80%)
- Validation: 3,000 images (10%)
- Test: 3,000 images (10%)

## Model Architecture

- **Base model:** VGG-16 pretrained on ImageNet
- **Variants:**
  - Freeze all convolutional layers → Train classifier only
  - Fine-tune the last convolutional block + classifier

## Training Configurations

| Configuration                     | Learning Rate | Fine-Tuning Strategy       |
|----------------------------------|---------------|----------------------------|
| Model 1                          | 0.001         | Freeze all conv. layers    |
| Model 2                          | 0.0001        | Freeze all conv. layers    |
| Model 3                          | 0.001         | Fine-tune last block       |
| Model 4                          | 0.0001        | Fine-tune last block       |

Each model was trained for **10 epochs**.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | Train Time (s) |
|-------|----------|-----------|--------|----------|----------------|
| M1 (0.001, Freeze All) | 0.9397   | 0.9242    | 0.9323 | 0.9282   | 11047.56        |
| M2 (0.0001, Freeze All) | 0.9393   | 0.9261    | 0.9291 | 0.9276   | 1713.94         |
| M3 (0.001, Fine-Tune)   | **0.9650**   | **0.9615**    | **0.9546** | **0.9580**   | 1719.30         |
| M4 (0.0001, Fine-Tune)  | 0.9587   | 0.9557    | 0.9450 | 0.9503   | 1721.22         |

- Fine-tuning improved accuracy by ~2%.
- Best performance from Model 3: fine-tuning + higher learning rate.
- No major overfitting or imbalance was observed.

## Key Takeaways

- **Fine-tuning** significantly boosts performance over freezing all layers.
- **Learning rate** had minor effects, more pronounced in fine-tuning cases.
- Confusion matrices confirmed **balanced classification** between genders.
- All models reached **high accuracy (~94–96%)**.

## Notebook

You can view the original notebook here:  
📎 [Google Colab Notebook](https://drive.google.com/file/d/1YQtea41Jd9IApPdjg6MKlvqj56cCLrgB/view?usp=sharing)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vgg16-gender-classification.git
   cd vgg16-gender-classification
