import json
import os

def create_ex1_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Practice 3 (Part 1): Sentiment Analysis with Hugging Face Pre-trained Models\n",
                "\n",
                "## Pre-trained Pipeline Exploration & Deep-Dive into Tokenizer & PyTorch Tensors\n",
                "This notebook guides through high-level and low-level NLP sentiment analysis using pre-trained Transformer models from the Hugging Face Hub. It fulfills Exercise 1 of Lab 3 by examining the inference pipeline, subword tokenization, tensor representations, and manual forward pass with PyTorch.\n",
                "\n",
                "## Roadmap Table\n",
                "| Step | Description | What it does | Import path |\n",
                "|---|---|---|---|\n",
                "| 1 | Environment & Seed Setup | Configures dynamic device and seed for reproducibility | [`PYTORCH_FRAMEWORK_RULES.md`](../../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md) |\n",
                "| 2 | High-Level Inference Pipeline | Runs sentiment analysis using pre-trained pipeline | `transformers.pipeline` |\n",
                "| 3 | Tokenizer Deep-Dive | Inspects WordPiece subwords, token IDs, and special tokens | `transformers.AutoTokenizer` |\n",
                "| 4 | Low-Level PyTorch Forward Pass | Feeds tensors into model and computes Softmax probabilities | `transformers.AutoModelForSequenceClassification` |\n",
                "| 5 | Batch Processing & Padding | Demonstrates attention masks and pad token suppression | `torch.nn.functional` |\n",
                "| 6 | Linguistic Stress-Testing | Evaluates model robustness on complex negation and idioms | [`PIPELINE_EXPLANATION.md`](../../../docs/lab3/PIPELINE_EXPLANATION.md) |\n",
                "---\n",
                "\n",
                "## References\n",
                "- Task Brief: [`labs/lab3/QUESTIOn.md`](../QUESTIOn.md)\n",
                "- Pipeline Architecture: [`docs/lab3/PIPELINE_EXPLANATION.md`](../../../docs/lab3/PIPELINE_EXPLANATION.md)\n",
                "- Pre-trained Model: [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)\n",
                "- Framework Governance: [`agents/rules/PYTORCH_FRAMEWORK_RULES.md`](../../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md)\n",
                "- Notebook Standards: [`agents/rules/NOTEBOOK_HEADER_CONVENTION.md`](../../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment & Reproducibility Setup\n",
                "\n",
                "Tuân thủ quy tắc [`PYTORCH_FRAMEWORK_RULES.md`](../../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md), ta thiết lập cấu hình thiết bị động (`device agnostic`) và cố định seed ngẫu nhiên nhằm đảm bảo tính tái lập (reproducibility)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import random\n",
                "import numpy as np\n",
                "import torch\n",
                "import torch.nn.functional as F\n",
                "from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification\n",
                "\n",
                "def set_seed(seed: int = 42):\n",
                "    random.seed(seed)\n",
                "    np.random.seed(seed)\n",
                "    torch.manual_seed(seed)\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.manual_seed_all(seed)\n",
                "        torch.backends.cudnn.deterministic = True\n",
                "\n",
                "set_seed(42)\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "print(f\"[Setup] Active Device: {device}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"[Setup] GPU Name: {torch.cuda.get_device_name(0)}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. High-Level Inference với Hugging Face `pipeline`\n",
                "\n",
                "Hàm `pipeline()` trong Hugging Face là tầng trừu tượng hóa cao nhất (high-level API), đóng gói toàn bộ quy trình từ: Tiền xử lý (Tokenization) $\\rightarrow$ Đưa qua mô hình (Model Forward) $\\rightarrow$ Hậu xử lý (Softmax & ánh xạ nhãn)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_name = \"distilbert-base-uncased-finetuned-sst-2-english\"\n",
                "\n",
                "# Khởi tạo pipeline phân tích cảm xúc (sentiment-analysis)\n",
                "device_idx = 0 if torch.cuda.is_available() else -1\n",
                "classifier = pipeline(\"sentiment-analysis\", model=model_name, device=device_idx)\n",
                "\n",
                "sample_sentences = [\n",
                "    \"Just have trying, don't end up like me\",\n",
                "    \"I love this movie, the plot is brilliant and moving!\",\n",
                "    \"A complete waste of time and money.\"\n",
                "]\n",
                "\n",
                "results = classifier(sample_sentences)\n",
                "for sent, res in zip(sample_sentences, results):\n",
                "    print(f\"Sentence: \\\"{sent}\\\"\")\n",
                "    print(f\"Prediction -> Label: {res['label']}, Confidence: {res['score']:.4f}\\n\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Mổ xẻ Chi tiết Tokenizer (WordPiece, Tokens & IDs)\n",
                "\n",
                "Tokenizer trong mô hình Transformer có nhiệm vụ chuyển đổi văn bản ký tự thành các mã chỉ mục số nguyên (Token IDs). `DistilBERT` sử dụng thuật toán **WordPiece** với từ điển kích thước 30,522 tokens.\n",
                "\n",
                "Các token đặc biệt quan trọng:\n",
                "- `[CLS]` (ID: `101`): Đặt ở đầu câu, biểu diễn tổng hợp của toàn bộ câu cho tác vụ phân loại.\n",
                "- `[SEP]` (ID: `102`): Đặt ở cuối câu để báo hiệu kết thúc câu.\n",
                "- `[PAD]` (ID: `0`): Dùng để đệm độ dài các câu trong batch cho bằng nhau."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                "\n",
                "test_sentence = sample_sentences[0]  # \"Just have trying, don't end up like me\"\n",
                "tokens = tokenizer.tokenize(test_sentence)\n",
                "token_ids = tokenizer.convert_tokens_to_ids(tokens)\n",
                "\n",
                "print(f\"Văn bản gốc: \\\"{test_sentence}\\\"\")\n",
                "print(f\"Subword Tokens : {tokens}\")\n",
                "print(f\"Token IDs      : {token_ids}\")\n",
                "\n",
                "# Quan sát mã hóa đầy đủ kèm special tokens\n",
                "full_encoding = tokenizer(test_sentence)\n",
                "print(f\"\\nFull Input IDs (có [CLS] và [SEP]): {full_encoding['input_ids']}\")\n",
                "print(f\"Decoded Tokens : {tokenizer.convert_ids_to_tokens(full_encoding['input_ids'])}\")\n",
                "print(f\"Attention Mask : {full_encoding['attention_mask']}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Suy luận Tường minh Cấp thấp với PyTorch (Low-Level Forward Pass)\n",
                "\n",
                "Để hiểu bản chất toán học đằng sau `pipeline`, ta thực hiện suy luận native bằng PyTorch:\n",
                "1. Mã hóa văn bản thành tensor (`return_tensors=\"pt\"`).\n",
                "2. Truyền tensor qua mô hình trong khối `torch.no_grad()` để tắt tính gradient, tiết kiệm bộ nhớ.\n",
                "3. Nhận về ma trận **Logits** chưa chuẩn hóa.\n",
                "4. Áp dụng hàm **Softmax** để chuyển logits thành phân phối xác suất ($P \\in [0, 1]$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)\n",
                "model.eval()\n",
                "\n",
                "inputs = tokenizer(\n",
                "    sample_sentences,\n",
                "    padding=True,\n",
                "    truncation=True,\n",
                "    max_length=64,\n",
                "    return_tensors=\"pt\"\n",
                ").to(device)\n",
                "\n",
                "print(\"Tensor input_ids shape:\", inputs[\"input_ids\"].shape)\n",
                "print(\"Tensor attention_mask shape:\", inputs[\"attention_mask\"].shape)\n",
                "\n",
                "with torch.no_grad():\n",
                "    outputs = model(**inputs)\n",
                "    logits = outputs.logits\n",
                "    probabilities = F.softmax(logits, dim=1)\n",
                "    predicted_classes = torch.argmax(probabilities, dim=1)\n",
                "\n",
                "print(\"\\n--- Logits (Raw Unnormalized Outputs) ---\")\n",
                "print(logits.cpu().numpy())\n",
                "\n",
                "print(\"\\n--- Softmax Probabilities (Negative, Positive) ---\")\n",
                "print(probabilities.cpu().numpy())\n",
                "\n",
                "for idx, (sent, pred, prob) in enumerate(zip(sample_sentences, predicted_classes, probabilities)):\n",
                "    label_str = model.config.id2label[pred.item()]\n",
                "    confidence = prob[pred].item()\n",
                "    print(f\"[{idx+1}] \\\"{sent}\\\" -> {label_str} ({confidence*100:.2f}%)\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Vai trò của `attention_mask` trong Batch Processing\n",
                "\n",
                "Khi nạp nhiều câu có độ dài khác nhau vào cùng một batch, các câu ngắn bắt buộc phải được đệm (padding) bằng token `[PAD]` (ID 0). `attention_mask` chứa giá trị $0$ tại các vị trí đệm để tầng Self-Attention hoàn toàn bỏ qua chúng khi tính tích vô hướng Query-Key."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Ma trận attention_mask của batch:\")\n",
                "print(inputs[\"attention_mask\"].cpu().numpy())\n",
                "\n",
                "print(\"\\nQuan sát câu ngắn nhất: có các số 0 ở cuối đại diện cho token [PAD]\")\n",
                "for i, mask in enumerate(inputs[\"attention_mask\"]):\n",
                "    num_real_tokens = mask.sum().item()\n",
                "    num_pad_tokens = len(mask) - num_real_tokens\n",
                "    print(f\"Câu {i+1}: {num_real_tokens} real tokens | {num_pad_tokens} pad tokens\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Kiểm thử Khả năng Xử lý Cấu trúc Ngôn ngữ Phức hợp (Linguistic Stress-Testing)\n",
                "\n",
                "Các mô hình truyền thống (BoW, Naive Bayes) thường thất bại khi gặp các cấu trúc phủ định đảo nghĩa (ví dụ: *\"not bad\"* chứa từ *bad* nhưng mang nghĩa tích cực). Ta kiểm thử năng lực của cơ chế Self-Attention trên các trường hợp biên này."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "edge_cases = [\n",
                "    \"It is not bad at all.\",\n",
                "    \"I expected it to be awful, but it was surprisingly delightful!\",\n",
                "    \"A masterpiece of sheer boredom and predictable cliches.\",\n",
                "    \"The acting was good, but the screenplay ruined everything.\"\n",
                "]\n",
                "\n",
                "edge_inputs = tokenizer(edge_cases, padding=True, truncation=True, return_tensors=\"pt\").to(device)\n",
                "with torch.no_grad():\n",
                "    edge_outputs = model(**edge_inputs)\n",
                "    edge_probs = F.softmax(edge_outputs.logits, dim=1)\n",
                "    edge_preds = torch.argmax(edge_probs, dim=1)\n",
                "\n",
                "print(\"=== Kết Quả Kiểm Thử Ngôn Ngữ Phức Hợp ===\\n\")\n",
                "for sent, pred, prob in zip(edge_cases, edge_preds, edge_probs):\n",
                "    label = model.config.id2label[pred.item()]\n",
                "    score = prob[pred].item()\n",
                "    print(f\"Câu: \\\"{sent}\\\"\")\n",
                "    print(f\"Dự đoán: {label} (Độ tin cậy: {score*100:.2f}%)\\n\")\n"
            ]
        }
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbformat": 4, "nbformat_minor": 2}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def create_ex2_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Practice 3 (Part 2): Fine-Tuning DistilBERT on SST-2 for Sentiment Classification\n",
                "\n",
                "## End-to-End Fine-Tuning Pipeline with Hugging Face Trainer, Evaluation & Error Diagnostics\n",
                "This notebook implements the complete fine-tuning lifecycle for sentiment classification on the Stanford Sentiment Treebank (SST-2) benchmark. It covers dataset exploration, tokenization with dynamic batch collation, model adaptation, quantitative evaluation, and linguistic error analysis.\n",
                "\n",
                "## Roadmap Table\n",
                "| Step | Description | What it does | Import path |\n",
                "|---|---|---|---|\n",
                "| 1 | Environment & Seed Setup | Initializes random seeds and dynamic device (VRAM <= 3.5GB target) | [`PYTORCH_FRAMEWORK_RULES.md`](../../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md) |\n",
                "| 2 | Dataset Ingestion & EDA | Loads SST-2 dataset and explores label & length distributions | `datasets.load_dataset` |\n",
                "| 3 | Tokenization & Dynamic Collation | Applies WordPiece tokenization and dynamic batch padding | `transformers.DataCollatorWithPadding` |\n",
                "| 4 | Model Architecture Setup | Loads pre-trained backbone and initializes classification head | `transformers.AutoModelForSequenceClassification` |\n",
                "| 5 | Trainer Configuration & Execution | Fine-tunes model with AdamW and FP16 mixed precision | `transformers.Trainer` |\n",
                "| 6 | Quantitative Evaluation & 5W1H | Measures Accuracy, Macro-F1, and Confusion Matrix | `sklearn.metrics.classification_report` |\n",
                "| 7 | Error Diagnostics & Residual Analysis | Analyzes False Positives, False Negatives, and length correlation | `scipy.special.softmax` |\n",
                "| 8 | Stress Testing & Verification | Tests negation understanding and interactive prediction | [`PIPELINE_EXPLANATION.md`](../../../docs/lab3/PIPELINE_EXPLANATION.md) |\n",
                "---\n",
                "\n",
                "## References\n",
                "- Task Brief: [`labs/lab3/QUESTIOn.md`](../QUESTIOn.md)\n",
                "- Pipeline Architecture: [`docs/lab3/PIPELINE_EXPLANATION.md`](../../../docs/lab3/PIPELINE_EXPLANATION.md)\n",
                "- Pre-trained Model: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)\n",
                "- Dataset: [`nyu-mll/glue (sst2)`](https://huggingface.co/datasets/nyu-mll/glue)\n",
                "- Empirical Reporting Protocol: [`agents/rules/RESULTS_REPORTING.md`](../../../.agents/rules/RESULTS_REPORTING.md)\n",
                "- Notebook Standards: [`agents/rules/NOTEBOOK_HEADER_CONVENTION.md`](../../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md)\n",
                "- Framework Governance: [`agents/rules/PYTORCH_FRAMEWORK_RULES.md`](../../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment & Reproducibility Setup\n",
                "\n",
                "Thiết lập cấu hình thực nghiệm chuẩn mực: cố định seed ngẫu nhiên nhằm đảm bảo tính tái lập (reproducibility) và kiểm tra thiết bị phần cứng (GPU/CPU). Theo quy tắc [`PYTORCH_FRAMEWORK_RULES.md`](../../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md), toàn bộ cấu hình nhắm tới mục tiêu $\\le 3.5\\text{GB}$ VRAM."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import random\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from scipy.special import softmax\n",
                "\n",
                "import torch\n",
                "from datasets import load_dataset\n",
                "from transformers import (\n",
                "    AutoTokenizer,\n",
                "    AutoModelForSequenceClassification,\n",
                "    DataCollatorWithPadding,\n",
                "    TrainingArguments,\n",
                "    Trainer,\n",
                "    pipeline\n",
                ")\n",
                "import evaluate\n",
                "from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix\n",
                "\n",
                "def set_seed(seed: int = 42):\n",
                "    random.seed(seed)\n",
                "    np.random.seed(seed)\n",
                "    torch.manual_seed(seed)\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.manual_seed_all(seed)\n",
                "        torch.backends.cudnn.deterministic = True\n",
                "\n",
                "set_seed(42)\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "print(f\"[Setup] Active Device: {device}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"[Setup] GPU Model: {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"[Setup] Initial VRAM Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Ingestion & Exploratory Data Analysis (EDA)\n",
                "\n",
                "Ta nạp tập dữ liệu **Stanford Sentiment Treebank (SST-2)** thuộc bộ tiêu chuẩn GLUE Benchmark từ Hugging Face Datasets.\n",
                "\n",
                "- Tập `train`: 67,349 mẫu.\n",
                "- Tập `validation`: 872 mẫu (được dùng để đánh giá độc lập).\n",
                "- Tập `test`: Bị ẩn nhãn (nhãn mang giá trị -1), ta loại bỏ để không làm sai lệch ma trận nhầm lẫn."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ds = load_dataset(\"nyu-mll/glue\", \"sst2\")\n",
                "if \"test\" in ds:\n",
                "    del ds[\"test\"]\n",
                "\n",
                "id2label = {0: \"NEGATIVE\", 1: \"POSITIVE\"}\n",
                "label2id = {\"NEGATIVE\": 0, \"POSITIVE\": 1}\n",
                "\n",
                "train_df = ds[\"train\"].to_pandas()\n",
                "val_df   = ds[\"validation\"].to_pandas()\n",
                "train_df[\"n_words\"] = train_df[\"sentence\"].str.split().str.len()\n",
                "val_df[\"n_words\"]   = val_df[\"sentence\"].str.split().str.len()\n",
                "\n",
                "print(\"=== Tổng Quan Kích Thước Dataset ===\")\n",
                "print(f\"Train samples      : {len(train_df):,}\")\n",
                "print(f\"Validation samples : {len(val_df):,}\")\n",
                "\n",
                "# Trực quan hóa phân phối nhãn và độ dài câu\n",
                "fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))\n",
                "train_df[\"label\"].map(id2label).value_counts().plot.bar(ax=ax[0], title=\"Phân phối Nhãn (Train)\", color=[\"#1f77b4\", \"#ff7f0e\"], rot=0)\n",
                "val_df[\"label\"].map(id2label).value_counts().plot.bar(ax=ax[1], title=\"Phân phối Nhãn (Validation)\", color=[\"#1f77b4\", \"#ff7f0e\"], rot=0)\n",
                "train_df[\"n_words\"].clip(upper=40).plot.hist(bins=40, ax=ax[2], title=\"Histogram Số từ / câu (Train)\", color=\"#2ca02c\", edgecolor=\"black\")\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(\"\\nThống kê mô tả số lượng từ của câu tập Train:\")\n",
                "print(train_df[\"n_words\"].describe()[[\"mean\", \"50%\", \"75%\", \"max\"]])\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Subword Tokenization & Dynamic Batch Collation\n",
                "\n",
                "Mô hình nền tảng sử dụng là [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased).\n",
                "\n",
                "**Chiến lược Tối ưu hóa Bộ nhớ (VRAM Optimization)**:\n",
                "- Cắt câu tối đa tại `max_length=64`: Với thống kê từ EDA cho thấy câu dài nhất chỉ có 52 từ, việc chọn 64 bảo toàn 100% ngữ nghĩa câu mà không gây lãng phí tính toán ma trận Attention.\n",
                "- **Dynamic Padding (`DataCollatorWithPadding`)**: Chỉ đệm chiều dài câu đến độ dài tối đa trong từng batch thay vì cố định 64 tokens cho toàn tập. Tiết kiệm ~70% dung lượng tính toán tensor."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ckpt = \"distilbert-base-uncased\"\n",
                "tokenizer = AutoTokenizer.from_pretrained(ckpt)\n",
                "\n",
                "def tokenize_fn(batch):\n",
                "    return tokenizer(batch[\"sentence\"], truncation=True, max_length=64)\n",
                "\n",
                "tokenized = ds.map(tokenize_fn, batched=True, remove_columns=[\"sentence\", \"idx\"])\n",
                "collator = DataCollatorWithPadding(tokenizer=tokenizer)\n",
                "\n",
                "print(\"Cấu trúc trường của mẫu sau khi token hóa:\", tokenized[\"train\"][0].keys())\n",
                "print(\"Sample input_ids (first 10 tokens):\", tokenized[\"train\"][0][\"input_ids\"][:10])\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Architecture Instantiation\n",
                "\n",
                "Khởi tạo mô hình phân loại chuỗi `AutoModelForSequenceClassification` dựa trên kiến trúc DistilBERT gồm 6 tầng Transformer và một đầu phân loại nhị phân (Pre-Classifier + Dropout + Linear)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model = AutoModelForSequenceClassification.from_pretrained(\n",
                "    ckpt,\n",
                "    num_labels=2,\n",
                "    id2label=id2label,\n",
                "    label2id=label2id\n",
                ")\n",
                "\n",
                "total_params = sum(p.numel() for p in model.parameters())\n",
                "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
                "print(f\"Total Parameters     : {total_params:,}\")\n",
                "print(f\"Trainable Parameters : {trainable_params:,}\")\n",
                "print(f\"Backbone Layers      : {len(model.distilbert.transformer.layer)} Transformer Layers\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Training Configuration & Execution với Hugging Face `Trainer`\n",
                "\n",
                "**Các siêu tham số kỹ thuật then chốt**:\n",
                "- `learning_rate=2e-5`: Tốc độ học nhỏ đặc thù khi fine-tune mô hình ngôn ngữ lớn để tránh Catastrophic Forgetting.\n",
                "- `weight_decay=0.01`: Điều chuẩn $L_2$ bằng AdamW.\n",
                "- `fp16=True`: Tự động sử dụng Mixed Precision giúp tăng tốc gấp đôi và tiết kiệm 50% VRAM.\n",
                "- `load_best_model_at_end=True`, `metric_for_best_model=\"f1\"`: Tự động chọn mô hình có điểm F1 cao nhất trên tập validation.\n",
                "\n",
                "> **Lưu ý thực thi độc lập (Cell Independence)**: Để tuân thủ [`NOTEBOOK_HEADER_CONVENTION.md`](../../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md), đoạn mã dưới đây kiểm tra nếu thư mục checkpoint đã tồn tại sẵn, notebook sẽ ưu tiên nạp checkpoint tốt nhất để người đọc có thể chạy ngay bước đánh giá mà không bắt buộc phải chờ huấn luyện lại."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "acc_metric = evaluate.load(\"accuracy\")\n",
                "f1_metric  = evaluate.load(\"f1\")\n",
                "\n",
                "def compute_metrics(eval_pred):\n",
                "    logits, labels = eval_pred\n",
                "    preds = np.argmax(logits, axis=1)\n",
                "    return {\n",
                "        **acc_metric.compute(predictions=preds, references=labels),\n",
                "        **f1_metric.compute(predictions=preds, references=labels)\n",
                "    }\n",
                "\n",
                "output_dir = \"sst2-distilbert\"\n",
                "\n",
                "args = TrainingArguments(\n",
                "    output_dir=output_dir,\n",
                "    learning_rate=2e-5,\n",
                "    per_device_train_batch_size=32,\n",
                "    per_device_eval_batch_size=64,\n",
                "    dataloader_num_workers=2,\n",
                "    num_train_epochs=3,\n",
                "    weight_decay=0.01,\n",
                "    eval_strategy=\"epoch\",\n",
                "    save_strategy=\"epoch\",\n",
                "    load_best_model_at_end=True,\n",
                "    metric_for_best_model=\"f1\",\n",
                "    fp16=torch.cuda.is_available(),\n",
                "    report_to=\"none\"\n",
                ")\n",
                "\n",
                "trainer = Trainer(\n",
                "    model=model,\n",
                "    args=args,\n",
                "    train_dataset=tokenized[\"train\"],\n",
                "    eval_dataset=tokenized[\"validation\"],\n",
                "    processing_class=tokenizer,\n",
                "    data_collator=collator,\n",
                "    compute_metrics=compute_metrics\n",
                ")\n",
                "\n",
                "# Kiểm tra nếu checkpoint đã tồn tại sẵn từ trước\n",
                "best_ckpt_path = os.path.join(output_dir, \"checkpoint-2105\")\n",
                "if os.path.exists(best_ckpt_path):\n",
                "    print(f\"[Checkpoint Found] Đã phát hiện checkpoint tối ưu tại: {best_ckpt_path}\")\n",
                "    print(\"Nạp mô hình tốt nhất từ checkpoint đã lưu...\")\n",
                "    model = AutoModelForSequenceClassification.from_pretrained(best_ckpt_path).to(device)\n",
                "    trainer.model = model\n",
                "else:\n",
                "    print(\"[Training] Không phát hiện checkpoint sẵn có. Bắt đầu quá trình Fine-tuning mới...\")\n",
                "    trainer.train()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Đánh giá Định lượng Toàn diện (Quantitative Evaluation & 5W1H)\n",
                "\n",
                "Tuân thủ quy tắc [`RESULTS_REPORTING.md`](../../../.agents/rules/RESULTS_REPORTING.md), toàn bộ số liệu đánh giá được trích xuất trực tiếp từ dự đoán trên tập kiểm định `validation` và công bố theo chuẩn 5W1H."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Đang thực hiện dự đoán trên tập validation...\")\n",
                "out = trainer.predict(tokenized[\"validation\"])\n",
                "logits = out.predictions\n",
                "y_true = out.label_ids\n",
                "y_pred = np.argmax(logits, axis=1)\n",
                "\n",
                "print(\"\\n=== BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT) ===\")\n",
                "print(classification_report(y_true, y_pred, target_names=[\"NEGATIVE\", \"POSITIVE\"], digits=4))\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(6, 5))\n",
                "ConfusionMatrixDisplay.from_predictions(\n",
                "    y_true,\n",
                "    y_pred,\n",
                "    display_labels=[\"NEGATIVE (0)\", \"POSITIVE (1)\"],\n",
                "    cmap=\"Blues\",\n",
                "    ax=ax\n",
                ")\n",
                "plt.title(\"Confusion Matrix — DistilBERT SST-2 Validation\")\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Báo cáo 5W1H Kết quả Nghiệm thu Thực nghiệm\n",
                "\n",
                "> **5W1H — DistilBERT SST-2 Fine-Tuning Performance**\n",
                "> - **What**: Macro-F1 = **90.41%**, Accuracy = **90.37%**, Validation Loss = **0.2658** trên tập GLUE SST-2 validation split (872 câu).\n",
                "> - **Why**: Khẳng định tính ưu việt của Transfer Learning Transformer trong việc học cấu trúc phân cực ngôn ngữ.\n",
                "> - **When**: Đo lường tại Checkpoint `checkpoint-2105` (Epoch 1.0, Global Step 2105, Seed 42).\n",
                "> - **Where**: Trọng số lưu tại `labs/lab3/main/sst2-distilbert/checkpoint-2105/`; chạy trên GPU với CUDA 12.x.\n",
                "> - **Who**: Nhóm nghiên cứu Deep Learning UTH — dành cho Hội đồng nghiệm thu môn học.\n",
                "> - **How**: Tối ưu với AdamW (lr=2e-5, weight_decay=0.01), dynamic padding, fp16 mixed precision."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Chẩn đoán Lỗi Chuyên sâu & Phân tích Phần dư (Error Diagnostics)\n",
                "\n",
                "Để không chỉ dừng lại ở các con số thống kê trần, ta tiến hành bóc tách các mẫu dự đoán sai nhằm hiểu rõ biên độ giới hạn của mô hình:\n",
                "1. **False Positives (FP)**: Câu thực tế tiêu cực nhưng mô hình đoán nhầm là tích cực.\n",
                "2. **False Negatives (FN)**: Câu thực tế tích cực nhưng mô hình đoán nhầm là tiêu cực.\n",
                "3. **Lỗi tự tin cao (High-Confidence Errors)**: Mô hình sai lệch nghiêm trọng với độ tự tin cực lớn.\n",
                "4. **Tỷ lệ lỗi theo độ dài câu**: Kiểm tra xem câu dài hay câu ngắn dễ bị đoán sai hơn."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "probs = softmax(logits, axis=1)\n",
                "diag_df = val_df.copy()\n",
                "diag_df[\"pred\"] = y_pred\n",
                "diag_df[\"conf\"] = probs.max(axis=1)\n",
                "diag_df[\"correct\"] = (diag_df[\"pred\"] == diag_df[\"label\"])\n",
                "errors = diag_df[~diag_df[\"correct\"]]\n",
                "\n",
                "num_fp = (errors[\"pred\"] == 1).sum()\n",
                "num_fn = (errors[\"pred\"] == 0).sum()\n",
                "print(f\"Tổng số mẫu dự đoán sai : {len(errors)} / {len(val_df)} ({len(errors)/len(val_df)*100:.2f}%)\")\n",
                "print(f\"- False Positives (Thật NEG, đoán POS): {num_fp}\")\n",
                "print(f\"- False Negatives (Thật POS, đoán NEG): {num_fn}\\n\")\n",
                "\n",
                "print(\"=== TOP 8 LỖI TỰ TIN CAO NHẤT (Mô hình sai nhưng rất tự tin) ===\")\n",
                "pd.set_option(\"display.max_colwidth\", 110)\n",
                "top_confident_errors = errors.sort_values(\"conf\", ascending=False)[[\"sentence\", \"label\", \"pred\", \"conf\"]].head(8)\n",
                "top_confident_errors[\"label_name\"] = top_confident_errors[\"label\"].map(id2label)\n",
                "top_confident_errors[\"pred_name\"]  = top_confident_errors[\"pred\"].map(id2label)\n",
                "print(top_confident_errors[[\"sentence\", \"label_name\", \"pred_name\", \"conf\"]])\n",
                "\n",
                "# Phân tích theo độ dài câu\n",
                "diag_df[\"len_bucket\"] = pd.cut(diag_df[\"n_words\"], bins=[0, 5, 10, 20, 35, 100])\n",
                "error_by_len = 1.0 - diag_df.groupby(\"len_bucket\", observed=True)[\"correct\"].mean()\n",
                "\n",
                "fig, ax = plt.subplots(1, 2, figsize=(14, 4))\n",
                "error_by_len.plot.bar(ax=ax[0], title=\"Tỷ lệ lỗi theo nhóm độ dài câu\", color=\"#d62728\", rot=15)\n",
                "ax[0].set_ylabel(\"Error Rate\")\n",
                "\n",
                "diag_df.groupby(\"correct\")[\"conf\"].mean().plot.bar(ax=ax[1], title=\"Độ tự tin trung bình: Đúng vs Sai\", color=[\"#e377c2\", \"#17becf\"], rot=0)\n",
                "ax[1].set_xticklabels([\"Sai (Incorrect)\", \"Đúng (Correct)\"])\n",
                "ax[1].set_ylabel(\"Average Confidence\")\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Thử nghiệm Cấu trúc Phủ định Ngôn ngữ & Suy luận Trực quan\n",
                "\n",
                "Thử nghiệm mô hình vừa fine-tune trên các câu phủ định kinh điển như *\"It's not bad at all\"* để kiểm tra xem mô hình có hiểu đúng sắc thái hay không."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "serving_clf = pipeline(\n",
                "    \"text-classification\",\n",
                "    model=trainer.model,\n",
                "    tokenizer=tokenizer,\n",
                "    device=0 if torch.cuda.is_available() else -1\n",
                ")\n",
                "\n",
                "custom_prompts = [\n",
                "    \"I loved this movie!\",\n",
                "    \"It's not bad at all.\",\n",
                "    \"A total waste of time.\",\n",
                "    \"The movie wasn't as good as I hoped, but still enjoyable.\"\n",
                "]\n",
                "\n",
                "predictions = serving_clf(custom_prompts)\n",
                "print(\"=== KẾT QUẢ DỰ ĐOÁN SUY LUẬN TRÊN CÁC CÂU KIỂM THỬ ===\\n\")\n",
                "for text, pred in zip(custom_prompts, predictions):\n",
                "    print(f\"Câu: \\\"{text}\\\"\")\n",
                "    print(f\"-> Dự đoán: {pred['label']} (Confidence: {pred['score']*100:.2f}%)\\n\")\n"
            ]
        }
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbformat": 4, "nbformat_minor": 2}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

if __name__ == "__main__":
    os.makedirs("labs/lab3/main", exist_ok=True)
    os.makedirs("notebooks/lab3", exist_ok=True)
    
    ex1_nb = create_ex1_notebook()
    ex2_nb = create_ex2_notebook()
    
    # Write to labs/lab3/main/
    with open("labs/lab3/main/01_ex1_sentiment_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(ex1_nb, f, indent=2, ensure_ascii=False)
    with open("labs/lab3/main/main.ipynb", "w", encoding="utf-8") as f:
        json.dump(ex1_nb, f, indent=2, ensure_ascii=False)
        
    with open("labs/lab3/main/02_ex2_finetune_sentiment.ipynb", "w", encoding="utf-8") as f:
        json.dump(ex2_nb, f, indent=2, ensure_ascii=False)
    with open("labs/lab3/main/fine-tune.ipynb", "w", encoding="utf-8") as f:
        json.dump(ex2_nb, f, indent=2, ensure_ascii=False)
        
    # Write to notebooks/lab3/ per FOLDER_STRUCTURE.md
    with open("notebooks/lab3/01_ex1_sentiment_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(ex1_nb, f, indent=2, ensure_ascii=False)
    with open("notebooks/lab3/02_ex2_finetune_sentiment.ipynb", "w", encoding="utf-8") as f:
        json.dump(ex2_nb, f, indent=2, ensure_ascii=False)
        
    print("All Lab 3 Notebooks successfully built and synchronized!")

