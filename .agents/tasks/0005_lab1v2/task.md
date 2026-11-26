---
title: Lab 1 - FashionMNIST Classification
status: In Progress
owner: notworLe
pipeline: MODEL_PIPELINE.md
---

# Objective

Build a complete Deep Learning pipeline for image classification using FashionMNIST.

The goal is not merely to obtain a high accuracy model, but to understand every stage of a standard Deep Learning workflow, from problem definition to evaluation and reporting.

---

# Background

This laboratory belongs to the Deep Learning course. [`QUESTION.md`](../../../labs/lab1/QUESTION.md).

The exercise focuses on understanding:

- how image data flows through the pipeline
- how tensors change shape
- how preprocessing affects the data
- how a neural network learns
- how training and evaluation are performed
- how engineering documents support reproducibility

This project follows the standard AI Development Pipeline described in `MODEL_PIPELINE.md`.

---

# Input

Dataset

- FashionMNIST

Provided by

- torchvision.datasets.FashionMNIST

Input sample

- grayscale image
- size: 28×28
- tensor shape:
  (1, 28, 28)

Target

- 10 clothing categories

---

# Business Requirement

Although this is an academic exercise rather than a production system, the project should simulate a real AI engineering workflow.

The resulting pipeline should be reusable for future image classification projects.

---

# Problem Definition

Task: Image Classification

Input: FashionMNIST image

Output: Predicted clothing class

Baseline: Logistic Regression

Improved Model: Multi-Layer Perceptron (MLP)

Evaluation Metric

- Accuracy
- Loss
- F1 Score
- confusion matrix

---

# Requirements

## Documentation

Design the complete modeling pipeline following `MODEL_PIPELINE.md`.

Documents should explain

- data flow
- tensor shapes
- preprocessing
- model architecture
- training pipeline
- evaluation pipeline

---

## Implementation

Implement

- Dataset loading
- Data preprocessing
- DataLoader
- Baseline model
- MLP model
- Training pipeline
- Evaluation pipeline
- Inference pipeline
- Model saving/loading

---

## Experiment

Perform experiments including

- hidden dimensions
- learning rate
- optimizer
- batch size
- epoch

Record every experiment.

---

## Visualization

Generate

- loss curve
- accuracy curve
- predicted vs actual images

---

## Analysis

Compare

Baseline

vs

MLP

Analyze

- convergence
- training stability
- overfitting
- advantages
- limitations

---

| ID  | Task                                   | Pipeline Layer         | Output          |
| --- | -------------------------------------- | ---------------------- | --------------- |
| T1  | Design modeling pipeline documentation | Documentation          | modeling.md     |
| T2  | Explore FashionMNIST dataset           | Data Collection + EDA  | notebook        |
| T3  | Implement preprocessing                | Data Preprocessing     | transforms.py   |
| T4  | Implement Logistic Regression baseline | Model Engineering      | logistic.py     |
| T5  | Train baseline                         | Training               | checkpoint      |
| T6  | Evaluate baseline                      | Evaluation             | metrics         |
| T7  | Implement MLP                          | Model Engineering      | mlp.py          |
| T8  | Hyperparameter tuning                  | Continuous Improvement | experiment logs |
| T9  | Compare Baseline vs MLP                | Evaluation             | report          |
| T10 | Write final report                     | Documentation          | report.md       |

## Evalutation

Evaluate the main model is mlp

# Deliverables

Documentation

- docs/pipelines/modeling.md
- docs/models/logistic_regression.md
- docs/models/mlp.md

Scripts

- train.py
- evaluate.py
- inference.py

Notebook

- experiment.ipynb

Report

- experiment_report.md

Artifacts

- trained models
- loss figures
- prediction figures

---

# Acceptance Criteria

The project is considered complete when

- the entire pipeline is documented clearly
- tensor shapes are explained throughout the pipeline
- both baseline and MLP converge successfully
- training, evaluation, and inference scripts run correctly
- experiment results are reproducible
- report explains why MLP performs better than the baseline
- documentation is concise, consistent, and presentation-ready

Nếu thiếu thông tin, AI phải liệt kê rõ:

- assumption 1
- assumption 2

và yêu cầu kỹ sư xác nhận.

script sinh ra ở ROOT/src/vibe_scrtips/


không xem bài tập chỉ là "viết model MLP" mà xem nó như một dự án AI thu nhỏ, đi qua đầy đủ các lớp của pipeline từ yêu cầu → dữ liệu → mô hình → huấn luyện → đánh giá → báo cáo, đồng thời vẫn giữ phạm vi vừa đủ cho một bài thực hành học Deep Learning.