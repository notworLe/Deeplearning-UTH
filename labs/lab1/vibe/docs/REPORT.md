# 1. Abstract

Sau khi có kết qua evaluate, khi tóm tắt tại đây

# 2. Define
## 2.1 Define context
**Name:** So sánh hiệu năng CNN, MLP, Logistic trên bài toán phân loại FashionMNIST bằng PyTorch.  
**Context:** Bài thực hành Deeplearning với thư viện pytorch.  
**Target:**
- **Main:** Làm quen PyTorch, tự xây model.  
- **Extra:** Hoàn thành các yêu cầu của đề.   

**Constraint:**
- **Công bằng thí nghiệm:** cùng optimizer, lr, batch size, số epoch và seed cho cả 3 mô hình
- MLP và CNN có số tham số cùng bậc.    

## Define problem
**Input:** ảnh xám 28×28.  
**Output:** Danh sách nhãn tổng 10, báo cáo phân tích.  
**Metric:** Accuracy, số tham số, epoch, confusion matrix.    
**Baseline:** Logistic.  
**Experiments:** Cùng 1 seed, Augmentation.
  
# Dataset
## Collection
**Source:** FashionMNIST của Zalando, tải qua torchvision.datasets.  
**Metadata:** 70k ảnh: 60k train, 10k test. Mỗi ảnh xám 28×28, giá trị pixel 0-255.  
**Label names:**  10 lớp (0 T-shirt/top ... 9 Ankle boot) 
- **Input:** 70k ảnh xám 28×28 (uint8, 0-255), 10 lớp cân bằng, vật thể căn giữa khung gần như đều.
- **Output:** 3 tập train 55k / val 5k / test 10k; mean và std tính từ train.
- **Target:** có dữ liệu đã chia đúng, hiểu được cấu trúc và độ khó của dữ liệu.
- **Solution:** split stratified, seed 248; EDA chỉ trên train.
- **Limitation:** ảnh nhỏ, nền sạch, chỉ một dataset nên kết luận khó suy rộng; test 10k có sai số vài phần mười điểm phần trăm.

## Cleaning
Là bộ dữ liệu đã được làm sạch. Nên ta có thể bỏ qua bước data cleaning.  


## Splitting
**Train:** 55.000.  
**Validation:** 5.000  
**Test:** 10.000

## EDA
```text
               ┌────────────────────────────────────────────────────────┐
               │              LENS 1: GLOBAL TOPOLOGY                   │
               │  df.shape, df.dtypes, memory footprint, null counts    │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │          LENS 2: UNIVARIATE DISTRIBUTIONS              │
               │  Mean, Median, Skewness, Kurtosis, Histograms, KDE     │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │         LENS 3: MULTIVARIATE INTERACTIONS              │
               │  Correlation heatmaps, Pairplots, Mutual Information   │
               └────────────────────────────────────────────────────────┘
```
Lens 1: Global structure  
Lens 2: Distributions  
Lens 3: Visual & class separability  
→ Kết luận và quyết định cho bước tiếp theo  

# Model selection
## Lý do chọn mô hình
- Logistic: mô hình tuyến tính đơn giản nhất, làm mốc sàn (baseline)
- MLP: ra trước CNN; thêm tầng ẩn phi tuyến để vượt giới hạn tuyến tính của
  Logistic; nhưng flatten ảnh nên mất cấu trúc không gian
- CNN: bắt buộc theo bài học; phát triển sau MLP để khai thác cấu trúc
  không gian 

## Architecture
```text
Logistic:  [B,1,28,28] → Flatten [B,784] → Linear(784→10) [B,10]
           Tham số: 7.850

MLP:       [B,1,28,28] → Flatten [B,784]
           → Linear(784→128) → ReLU → Linear(128→64) → ReLU
           → Linear(64→10) [B,10]
           Tham số: 109.386

CNN:       [B,1,28,28]
           → Conv(1→16, 3×3, pad 1) → ReLU → MaxPool2   [B,16,14,14]
           → Conv(16→32, 3×3, pad 1) → ReLU → MaxPool2  [B,32,7,7]
           → Flatten [B,1568] → Linear(1568→64) → ReLU
           → Linear(64→10) [B,10]
           Tham số: 105.866
```

# Data processing
## Scaling
- ToTensor: 0-255 → [0,1], tensor 1×28×28
- Normalize: tính từ train, áp cho val và test
- Dùng chung cho cả 3 mô hình
- Thứ tự: augmentation → ToTensor → Normalize

## Augmentation
- Thí nghiệm: flip ngang, crop/dịch nhẹ, xoay nhỏ; chỉ trên train

## DataLoader
- batch size: 32, shuffle (train), seed: 248, worker: 8.


# Training
## Loss
- CrossEntropyLoss

## Optimizer
- Loại: Adam, lr

## Cấu hình huấn luyện
- Batch size 32, số epoch, seed 248, device (CPU/GPU)

## Vòng lặp
- forward → loss → zero_grad → backward → step
- Mỗi epoch: đánh giá trên validation (model.eval(), no_grad)

## Theo dõi & lưu
- Log: train/val loss, accuracy, thời gian/epoch
- Checkpoint: lưu model có val loss/accuracy tốt nhất


# Baseline
## Purpose
Tạo mốc sàn để biết MLP và CNN cải thiện bao nhiêu. Baseline không cần tốt, cần đơn giản và dễ giải thích.
## Architecture
Flatten [B,784] → Linear(784→10) [B,10]    (7.850 tham số, trả logits)

# Evaluate
## Metric
Accuracy.
Precision/recall/F1 theo từng lớp.
Confusion matrix.
Số tham số, epoch, kích thước file model, thời gian suy luận.
## Đường cong học
## Phân tích theo lớp
## Phân tích lỗi (confusion matrix, ảnh đúng/sai)
## Hạn chế và kết luận


# Error analysis
