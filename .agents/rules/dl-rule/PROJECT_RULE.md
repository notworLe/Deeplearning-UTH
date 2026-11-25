---
title: Project Rules
doc_type: Project Standard
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-23
updated: 2026-11-23
version: 1.0.0
status: approved
---

# Project Rules

## Overview

This document defines the engineering rules, repository conventions, and development standards for this project.

Every engineer and AI Agent must follow these rules before making any modification.

If a task conflicts with these rules, the agent must stop and request clarification instead of making assumptions.

---

# Repository Architecture

## Objective

Understand how the repository is organized.

The repository should be modular.

Each module has a single responsibility.

Examples

- data
- models
- training
- evaluation
- inference
- optimization
- serving

Do not place unrelated logic into the same module.

---

# Repository Structure

> **Lưu ý (Template):** Cấu trúc và các module `src/...` bên dưới là **mẫu chuẩn tham khảo** cho dự án ML/DL end-to-end. Khi áp dụng cho một repo cụ thể (ví dụ bài lab, research, hoặc dự án không có serving), hãy cập nhật lại cây thư mục thực tế bên dưới và **chỉ dùng các module mà dự án thật sự cần**.

```text
ROOT/
├── experiments/
├── docs/
├── data/
├── src/
├── tests/
├── scripts/
├── models/
└── configs/
```

(đặt tree của project cụ thể ở đây khi áp dụng sang repo dự án)

---

# Module Responsibilities

Each module has a clear responsibility.

## experiments/

Responsible for

- Experiment run results and metrics
- Benchmark comparisons and plots
- Per-run configuration snapshots
- Exploratory prototypes / notebooks (if any)

Must NOT

- Store production source code
- Replace base configurations in `configs/`

---

## docs/

Responsible for

- Project documentation
- Architecture
- Pipelines
- Module documentation
- Research notes and technical reports

Must NOT

- Store executable source code

---

## tests/

Responsible for

- Unit tests
- Integration tests
- Smoke tests

Must NOT

- Implement production features

---

## data/

Responsible for

- Raw datasets
- Interim datasets
- Processed datasets
- Dataset metadata

Must NOT

- Contain business logic
- Commit large dataset files directly to Git (use `.gitignore`, DVC, or external storage)

---

## configs/

Responsible for

- Base and reusable project configurations
- Model, data, and training hyperparameters
- Environment settings

Must NOT

- Contain executable logic
- Store plaintext secrets or credentials

---

## scripts/

Responsible for

- CLI entrypoints (e.g., `train.py`, `eval.py`, `infer.py`, `export.py`)
- Data preparation and automation scripts

Must NOT

- Implement reusable core logic (keep core modules inside `src/`)

---

## models/

Responsible for

- Trained checkpoints
- Exported models (e.g., ONNX, TensorRT)
- Model artifacts

Must NOT

- Contain source code
- Commit large binary weights directly to Git (use `.gitignore`, DVC, or Model Registry)

---

## src/data

Responsible for

- Dataset
- DataLoader
- Transform
- Split
- Validation

Must NOT

- Train model
- Evaluate model

---

## src/models

Responsible for

- Network architecture
- Loss
- Model factory

Must NOT

- Load dataset
- Train model

---

## src/training

Responsible for

- Training loop
- Checkpoint
- Optimizer
- Scheduler
- Callback

Must NOT

- Perform inference
- Compute deployment benchmark

---

## src/evaluation

Responsible for

- Metrics
- Error Analysis
- Visualization
- Benchmark Evaluation

---

## src/inference

Responsible for

- Prediction
- Preprocess
- Postprocess

Must NOT

- Update model parameters

---

## src/optimization

Responsible for

- ONNX Export
- TensorRT
- Quantization
- Pruning

---

## src/serving

Responsible for

- FastAPI
- Request schema
- Response schema
- Runtime inference

---

# Configuration Rules

All configurable values must come from configuration files.

Never hardcode

- learning rate
- batch size
- epochs
- dataset path
- model path

Configuration should remain outside business logic.

---

# Data Rules

Raw data must remain immutable.

Pipeline

Raw

↓

Interim

↓

Processed

Do not manually modify processed datasets.

Dataset generation must be reproducible.

---

# Tensor Contract

Every module should preserve tensor contracts.

Examples

Input

(B,3,H,W)

↓

Backbone

↓

Feature Maps

↓

Head

↓

Output

Tensor shape assumptions should be validated whenever appropriate.

---

# Coding Standards

General principles

- Single Responsibility Principle
- Modular Design
- Config-first
- Reusable components
- Explicit over implicit

Avoid

- duplicated logic
- magic numbers
- hardcoded paths
- hidden side effects

---

# Documentation Rules

Every complex module should include

README.md

describing

- Purpose
- Responsibilities
- Public API
- Dependencies
- Usage

Documentation should evolve together with implementation.

---

# Testing Rules

Every new feature should be verifiable.

Typical validation includes

- Smoke Test
- Functional Test
- Forward Pass
- Unit Test (when appropriate)

Do not merge unverified implementation.

---

# Dependency Rules

Before adding a new dependency,

consider

- maintenance cost
- project consistency
- existing alternatives

Avoid introducing unnecessary libraries.

---

# Repository Boundaries

The agent must not

- reorganize repository structure without approval
- rename public modules without approval
- introduce new architectural patterns arbitrarily
- modify unrelated modules

---

# Decision Priority

When multiple rules conflict, follow this order

1. `TASK.md`
2. Project Rules (`PROJECT_RULE.md`)
3. Pipeline Documentation (`MODEL_PIPELINE.md` / `DEPLOY_MODEL.md`)
4. Module `README.md`
5. General Engineering Best Practices

If the conflict cannot be resolved,

STOP

Ask the engineer for clarification.

---

# Relationship with Other Documents

Project Rules defines how the repository is organized.

It does not describe

- business requirements
- task objectives
- execution plans
- implementation history

Those responsibilities belong to

- Task
- Planning
- Execution
- Logging