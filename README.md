# PyTorch Journey

A structured, hands-on learning path to master PyTorch — from tensor basics to building real deep learning models.

---

## Repository Structure

```
pytorch-journey/
├── tensor-fundamentals/          # Core tensor operations & exercises
│   ├── 01_creating_tensors.ipynb
│   ├── 02_creating_tensors.ipynb
│   ├── 03_creating_tensors.ipynb
│   ├── 04_tensor_indexing.ipynb
│   ├── 05_tensor_slicing.ipynb
│   ├── 06_tensors_index_slice_practice.ipynb
│   ├── 07_tensors_reshaping.ipynb
│   ├── 08_tensors_squeeze_unsqueeze.ipynb
│   ├── 09_tensors_transpose.ipynb
│   ├── 10_tensors_permute.ipynb
│   ├── 11_tensors_multiplication.ipynb
│   ├── 12_tensors_broadcasting.ipynb
│   ├── 13_aggregate_methods.ipynb
│   ├── 14_tensors_exercises.ipynb
│   └── tensor-exercises/
│
├── models/
│   ├── regression/               # Linear regression models
│   │   ├── basic_linear_regression_model.ipynb
│   │   ├── single_linear_regression_model.ipynb
│   │   ├── multi_linear_regression_model.ipynb
│   │   ├── multi_weight_linear_regression_model.ipynb
│   │   ├── car_price_prediction_model.ipynb
│   │   ├── house_price_prediction_model.ipynb
│   │   ├── salary_prediction_model.ipynb
│   │   └── test_linear_regression_model.ipynb
│   │
│   ├── classification/           # Classification models
│   │   ├── non_linear_classification_model.ipynb
│   │   └── helpers/
│   │
│   └── concepts/                 # Core ML concepts
│       └── normalization/
│           ├── normalization-model.ipynb
│           └── unnormalization-model.ipynb
│
├── homework.md
├── questions.md
└── resources.md
```

---

## What's Covered

### Tensor Fundamentals
Everything you need to work confidently with PyTorch tensors:
- Creating tensors in multiple ways
- Indexing, slicing, and reshaping
- Squeeze, unsqueeze, transpose, and permute
- Matrix multiplication and broadcasting
- Aggregate methods (min, max, mean, sum, etc.)
- Hands-on exercises to reinforce concepts

### Regression Models
End-to-end regression model implementations:
- **Basic & Single Linear Regression** — fundamentals of gradient descent and loss
- **Multi-feature Regression** — predicting with multiple input features
- **Real-world datasets** — car price, house price, and salary prediction

### Classification Models
- **Non-linear Classification** — using activation functions to solve non-linear decision boundaries

### Concepts
- **Normalization vs Unnormalization** — understanding why and how to normalize inputs for stable training

---

## Learning Path

1. **Tensor Fundamentals** — start here; get comfortable with the building block of all PyTorch code
2. **Regression Models** — learn the training loop with simple, interpretable problems
3. **Concepts (Normalization)** — understand data preprocessing before moving to classification
4. **Classification Models** — apply everything to non-linear problems

---

## Getting Started

### Prerequisites
```bash
pip install torch torchvision jupyter matplotlib numpy pandas
```

### Run Notebooks
```bash
jupyter notebook
```
Navigate to any folder and open the `.ipynb` files in order.

---

## Resources

See [`resources.md`](./resources.md) for curated links, cheat sheets, and reference material.  
See [`homework.md`](./homework.md) for practice exercises and supplementary tasks.  
See [`questions.md`](./questions.md) for conceptual questions to test your understanding.

---

## Goal

By the end of this repo, you should be able to:

- Manipulate tensors with full confidence
- Implement linear and non-linear models from scratch
- Understand the full PyTorch training loop
- Apply data preprocessing (normalization) correctly
- Build and extend your own deep learning models
