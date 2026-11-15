# Abstract

# Define
## Define context
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


# Training
## Loss
- CrossEntropyLoss

## Optimizer
- Loại: Adam, lr, weight decay

## Cấu hình huấn luyện
- Batch size 32, số epoch, seed 248, device (CPU/GPU)

## Vòng lặp
- forward → loss → zero_grad → backward → step
- Mỗi epoch: đánh giá trên validation (model.eval(), no_grad)

## Theo dõi & lưu
- Log: train/val loss, accuracy, thời gian/epoch
- Checkpoint: lưu model có val loss/accuracy tốt nhất

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

# Baseline

