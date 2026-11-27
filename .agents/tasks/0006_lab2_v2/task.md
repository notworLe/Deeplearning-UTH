---

title: Lab 2 - Hands-on practice with pre-trained neural network architectures
status: In Progress
owner: notworLe
pipeline: MODEL_PIPELINE.md
created 2026-11-21
updated 2026-11-25
last_updated_by notworLe
version
reviewers

---

# Objective

Build a complete Deep Learning pipeline for image classification using pre-trained models from torchvision.models on Oxford-IIIT Pets.

## The goal is not merely to obtain a high accuracy model, but to understand how to load, adapt (freeze / unfreeze) and fine-tune pre-trained networks, and to measure whether transfer learning is worth it compared with training from scratch.

# Background

This laboratory belongs to the Deep Learning course. [`QUESTION.md`](../../../labs/lab2_v2/QUESTION.md).

The exercise focuses on understanding:

- how image data flows through the pipeline
- how tensors change shape
- how preprocessing affects the data
- how a neural network learns
- how training and evaluation are performed
- how engineering documents support reproducibility
- how to choose pretrained models
- how to use pretrain models for special tasks
- how to transfer learning as baseline
- how to fine tune models for special tasks
  Bài lab yêu cầu làm quen torchvision.models, thay classification head, freeze/unfreeze layer, dùng TensorBoard, thử model và hyperparameter khác nhau. Không có user thật, chỉ có mục tiêu học và so sánh.

This project follows the standard AI Development Pipeline described in `MODEL_PIPELINE.md`.

---

# Input

Dataset: Oxford-IIIT Pets

Provided by: torchvision.datasets.OxfordIIITPet (trainval + test)

Input sample

RGB image, resized to 224×224
tensor shape: (3, 224, 224)

Target

37 dog and cat breeds

Models

## resnet18, densenet121, vgg16 (weights="DEFAULT")

# Business Requirement

Although this is an academic exercise rather than a production system, the project should simulate a real AI engineering workflow.

Business question:

Can pet images be classified into 37 breeds automatically, and is a pre-trained model (transfer / fine-tune) better than training from scratch? Which architecture fits best?

The resulting pipeline should be reusable for future image classification projects.

---

# Problem Definition

Task: Image Classification (multi-class)

Input: Pet image (B, 3, ~200–400 px, ~200–400 px), ImageNet-normalized

Output: Logits (B, 37) → predicted breed

Data split

Train: trainval
Validation: 35% of test (stratified, fixed seed)
Test: remaining 65% of test, evaluated once

Baseline: ResNet18 trained from scratch

Compared Models

ResNet18 transfer (frozen backbone)
ResNet18 fine-tune (unfreeze last block)
DenseNet121 transfer
VGG16 transfer

Evaluation Metric

Accuracy
Loss
Per-class Precision / Recall / F1
Confusion matrix

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

Baseline (scratch)

vs

Transfer / Fine-tune

Analyze

convergence speed
training stability
overfitting
trainable parameters and training time
advantages
limitations

---

ID Task Pipeline Layer Output
T1 Design modeling pipeline documentation Documentation pipeline_overall.md
T2 Explore dataset, create 15% val split Data Collection + EDA notebook, split file
T3 Implement preprocessing Data Preprocessing transforms.py
T4 Inspect architectures, implement build_model Model Engineering model.py
T5 Train ResNet18 baseline (scratch) Training checkpoint
T6 Train ResNet18 transfer and fine-tune Training checkpoint
T7 Train DenseNet121 and VGG16 (transfer) Training checkpoint
T8 Hyperparameter experiments (lr, batch size) Continuous Improvement experiment logs
T9 TensorBoard monitoring Training Monitoring event logs
T10 Evaluate on val, then test once Evaluation metrics
T11 Compare runs and write final report Documentation experiment_report.md

## Evalutation

The main model is the run with the best validation accuracy. Test is evaluated once, after selection.

# Deliverables

Documentation

docs/pipelines/pipeline_overall.md
docs/pipelines/models/ (resnet18, densenet121, vgg16)
docs/pipelines/experiment.md
docs/pipelines/evaluation.md

Scripts

- train.py
- evaluate.py
- inference.py

Notebook

- experiment.ipynb

Report

- experiment_report.md

Artifacts

trained models (experiments/runs/<run_name>/)
confusion matrix and prediction figures

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

# Constraints

Full training is run by the engineer; the agent only runs Smoke / short Normal Tests.
No tuning on the test set.
No fabricated metrics before real runs.
No new dependencies beyond torch, torchvision, tensorboard.
Scripts are generated in ROOT/src/vibe_scrtips/.

# Assumptions (cần kỹ sư xác nhận)

Epoch, optimizer and learning rate defaults are proposed by the agent in the Plan.
finetune unfreezes layer4 for ResNet only; DenseNet and VGG run transfer only.
Best model is selected by validation accuracy.
