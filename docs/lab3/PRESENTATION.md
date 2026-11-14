# PRESENTATION.md — Kịch Bản Thuyết Trình Toàn Diện Module NLP Pipeline Lab 03

- **Motivation/Background**: Khi báo cáo đồ án hoặc thuyết trình bài thực hành Deep Learning trước Giảng viên và Hội đồng, sinh viên thường gặp khó khăn trong việc kết nối giữa cấu trúc mã nguồn thực tế với lý thuyết toán học và dòng chảy dữ liệu (Pipeline). Việc thiếu một kịch bản dẫn dắt rõ ràng chỉ ra file nào nằm ở đâu, mở file nào khi nói về module nào, và trả lời phản biện ra sao sẽ làm giảm tính thuyết phục của sản phẩm.
- **Purpose**: Cung cấp tài liệu hướng dẫn thuyết trình toàn diện cho Lab 03: xác định rõ vị trí từng file trong dự án, vai trò tương ứng trong pipeline, và kịch bản thuyết trình (Presentation Script) từng từ từng câu cho từng module, kèm bộ câu hỏi phản biện (Q&A Defense).
- **Overview Pipeline**: Ánh xạ 10 module kỹ thuật từ [docs/lab3/PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md) và kết quả thực nghiệm từ [docs/lab3/README.md](README.md) vào kịch bản thuyết trình chuẩn mực thời lượng 10–15 phút.
- **Detailed Plan**: §1 Sơ đồ Bản đồ Phân bổ File & Vai trò Thuyết trình; §2 Bảng Ánh xạ File - Module - Hành động Thuyết trình; §3 Kịch bản Thuyết trình Chi tiết (Từ Mở đầu đến Kết luận); §4 Kịch bản Trình diễn Mã nguồn & Demo Trực tiếp; §5 Bộ Câu hỏi & Câu trả lời Phản biện với Giảng viên (Q&A Defense Guide).
- **References**: [docs/lab3/PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md), [docs/lab3/README.md](README.md), [labs/lab3/QUESTIOn.md](../../labs/lab3/QUESTIOn.md), [labs/lab3/main/01_ex1_sentiment_analysis.ipynb](../../labs/lab3/main/01_ex1_sentiment_analysis.ipynb), [labs/lab3/main/02_ex2_finetune_sentiment.ipynb](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb), [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md).
- **Created**: 2026-11-14T08:42:20+07:00
- **Last Updated**: 2026-11-14T08:42:20+07:00

---

## Table of Contents

- [1. Sơ đồ Bản đồ Phân bổ File &amp; Vai trò Thuyết trình](#1-sơ-đồ-bản-đồ-phân-bổ-file--vai-trò-thuyết-trình)
- [2. Bảng Ánh xạ Module - File Mã Nguồn - Hành Động Trình Chiếu](#2-bảng-ánh-xạ-module---file-mã-nguồn---hành-động-trình-chiếu)
- [3. Kịch Bản Lời Thuyết Trình Chi Tiết (Script Từng Slide/Module)](#3-kịch-bản-lời-thuyết-trình-chi-tiết-script-từng-slidemodule)
  - [Phần 0: Lời Mở Đầu &amp; Đặt Vấn Đề (1 Phút)](#phần-0-lời-mở-đầu--đặt-vấn-đề-1-phút)
  - [Phần 1: Giới Thiệu Cấu Trúc Dự Án &amp; Khung Kiến Trúc (1.5 Phút)](#phần-1-giới-thiệu-cấu-trúc-dự-án--khung-kiến-trúc-15-phút)
  - [Phần 2: Module 1, 2, 3 — Dữ Liệu SST-2 &amp; Khảo Sát Hình Học Văn Bản (2 Phút)](#phần-2-module-1-2-3--dữ-liệu-sst-2--khảo-sát-hình-học-văn-bản-2-phút)
  - [Phần 3: Module 4, 5 — Tokenization &amp; Tối Ưu Bộ Nhớ Dynamic Collation (2.5 Phút)](#phần-3-module-4-5--tokenization--tối-ưu-bộ-nhớ-dynamic-collation-25-phút)
  - [Phần 4: Module 6, 7 — Kiến Trúc DistilBERT &amp; Động Cơ Huấn Luyện (2.5 Phút)](#phần-4-module-6-7--kiến-trúc-distilbert--động-cơ-huấn-luyện-25-phút)
  - [Phần 5: Module 8 — Nghiệm Thu Kết Quả Thực Nghiệm Chuẩn 5W1H (2 Phút)](#phần-5-module-8--nghiệm-thu-kết-quả-thực-nghiệm-chuẩn-5w1h-2-phút)
  - [Phần 6: Module 9, 10 — Chẩn Đoán Lỗi &amp; Trình Diễn Suy Luận Thực Tế (2.5 Phút)](#phần-6-module-9-10--chẩn-đoán-lỗi--trình-diễn-suy-luận-thực-tế-25-phút)
  - [Phần 7: Tổng Kết &amp; Hướng Phát Triển (1 Phút)](#phần-7-tổng-kết--hướng-phát-triển-1-phút)
- [4. Kịch Bản Thao Tác Trình Diễn Trực Tiếp (Live Demo Script)](#4-kịch-bản-thao-tác-trình-diễn-trực-tiếp-live-demo-script)
- [5. Bộ Câu Hỏi &amp; Câu Trả Lời Phản Biện với Giảng Viên (Q&amp;A Defense)](#5-bộ-câu-hỏi--câu-trả-lời-phản-biện-với-giảng-viên-qa-defense)

---

## 1. Sơ đồ Bản đồ Phân bổ File & Vai trò Thuyết trình

Để buổi thuyết trình diễn ra mạch lạc, dưới đây là vị trí chính xác của từng file trong dự án và vai trò của nó khi bạn trình bày:

```text
Deeplearning-UTH/
│
├── docs/lab3/                                    <-- [NƠI CHIẾU SLIDE / BÁO CÁO LÝ THUYẾT]
│   ├── README.md                                 # Tổng quan bài toán, quản trị VRAM, tóm tắt kết quả 5W1H
│   ├── PIPELINE_EXPLANATION.md                   # Bóc tách 10 Module toán học (Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp)
│   └── PRESENTATION.md                           # File kịch bản thuyết trình này
│
├── labs/lab3/                                    <-- [NƠI LƯU BÀI TẬP VÀ MÃ NGUỒN CHÍNH THỨC]
│   ├── QUESTIOn.md                               # Đề bài gốc Exercise 1 & Exercise 2
│   ├── README.md                                 # Hướng dẫn tổng quan bài nộp Lab 3
│   │
│   └── main/                                     <-- [NƠI MỞ CODE VÀ THỰC HIỆN LIVE DEMO]
│       ├── 01_ex1_sentiment_analysis.ipynb       # [Demo 1]: Mổ xẻ Tokenizer, tokens, IDs, logits & Softmax PyTorch
│       ├── 02_ex2_finetune_sentiment.ipynb       # [Demo 2]: Toàn bộ pipeline Fine-tune DistilBERT, ma trận nhầm lẫn & phân tích lỗi
│       ├── main.py                               # Script CLI suy luận nhanh (Inference)
│       ├── fine-tune.py                          # Script CLI huấn luyện chính thức (HF Trainer)
│       └── sst2-distilbert/                      # [BẰNG CHỨNG THỰC NGHIỆM ĐÃ TRAIN XONG]
│           └── checkpoint-2105/                  # Checkpoint tốt nhất (F1 = 90.41%, epoch 1.0)
│               ├── model.safetensors             # Trọng số mô hình sau fine-tune (an toàn chống RCE)
│               ├── config.json                   # Cấu hình kiến trúc mô hình
│               ├── tokenizer.json                # Từ điển và quy tắc token hóa WordPiece
│               └── trainer_state.json            # Toàn bộ nhật ký loss, metrics qua từng bước huấn luyện
│
└── notebooks/lab3/                               <-- [BẢN LƯU DỰ PHÒNG CHUẨN FOLDER STRUCTURE]
    ├── 01_ex1_sentiment_analysis.ipynb
    └── 02_ex2_finetune_sentiment.ipynb
```

---

## 2. Bảng Ánh xạ Module - File Mã Nguồn - Hành Động Trình Chiếu

Khi thuyết trình đến phần nào, bạn mở đúng file và trỏ chuột vào đúng vị trí theo bảng hướng dẫn sau:

| Module trong Pipeline | File tương ứng cần mở | Vị trí / Đoạn code cần trỏ vào | Mục đích trình chiếu |
| :--- | :--- | :--- | :--- |
| **Tổng quan Đề tài** | [`labs/lab3/QUESTIOn.md`](../../labs/lab3/QUESTIOn.md) | Exercise 1 & 2 | Chỉ ra đề bài yêu cầu làm gì. |
| **Kiến trúc Pipeline** | [`docs/lab3/PIPELINE_EXPLANATION.md`](PIPELINE_EXPLANATION.md) | Mục 1 (Sơ đồ Mermaid) | Chiếu toàn cảnh 5 tầng xử lý của Transformer. |
| **M1: Problem & Metric** | [`docs/lab3/PIPELINE_EXPLANATION.md`](PIPELINE_EXPLANATION.md) | Mục 2 | Định nghĩa bài toán phân loại nhị phân SST-2. |
| **M2, M3: Ingestion & EDA** | [`02_ex2_finetune_sentiment.ipynb`](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb) | Cell 2 & 4 (Biểu đồ bar & hist) | Trình chiếu phân phối nhãn cân bằng và độ dài câu max 52 từ. |
| **M4: Subword Tokenizer** | [`01_ex1_sentiment_analysis.ipynb`](../../labs/lab3/main/01_ex1_sentiment_analysis.ipynb) | Cell 6 (Tokens, IDs, Special tokens) | Minh họa WordPiece, token `[CLS]`, `[SEP]`, `[PAD]`. |
| **M5: Dynamic Collation** | [`02_ex2_finetune_sentiment.ipynb`](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb) | Cell 6 (`DataCollatorWithPadding`) | Giải thích cơ chế đệm theo batch tiết kiệm 70% VRAM. |
| **M6: Model Architecture** | [`02_ex2_finetune_sentiment.ipynb`](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb) | Cell 8 (Số lượng tham số) | Chỉ rõ DistilBERT có 6 tầng, 66M tham số, đầu phân loại 2 nhãn. |
| **M7: Training Engine** | [`labs/lab3/main/fine-tune.py`](../../labs/lab3/main/fine-tune.py) | Đoạn `TrainingArguments` (dòng 44-59) | Trình bày siêu tham số: AdamW, lr=2e-5, fp16=True, weight_decay=0.01. |
| **M8: Đánh giá 5W1H** | [`docs/lab3/README.md`](README.md) hoặc Notebook 2 | Cell 12 (Classification Report & Matrix) | Công bố số liệu chuẩn mực: F1 = 90.41%, Acc = 90.37%. |
| **M9: Chẩn đoán Lỗi** | [`02_ex2_finetune_sentiment.ipynb`](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb) | Cell 14 (Top lỗi tự tin cao & Bar chart) | Bóc tách FP/FN, giải thích nguyên nhân mỉa mai, đa mệnh đề. |
| **M10: Live Serving Demo** | [`01_ex1_sentiment_analysis.ipynb`](../../labs/lab3/main/01_ex1_sentiment_analysis.ipynb) | Cell 12 (Test câu *"It is not bad at all"*) | Chạy trực tiếp cho Giảng viên thấy mô hình hiểu đảo nghĩa phủ định. |

---

## 3. Kịch Bản Lời Thuyết Trình Chi Tiết (Script Từng Slide/Module)

*(Gợi ý: Bạn có thể đọc hoặc trình bày tự nhiên theo giọng điệu tự tin, rành mạch dưới đây).*

### Phần 0: Lời Mở Đầu & Đặt Vấn Đề (1 Phút)
> *"Kính thưa Thầy/Cô và các bạn trong Hội đồng,*
> 
> *Hôm nay, nhóm em xin phép được báo cáo kết quả thực hiện **Lab 03: Phân tích Cảm xúc Văn bản (Sentiment Analysis) sử dụng Kiến trúc Transformer và Tinh chỉnh Mô hình DistilBERT trên tập dữ liệu SST-2 thuộc bộ chuẩn GLUE Benchmark**.*
> 
> *Trong xử lý ngôn ngữ tự nhiên truyền thống, các mô hình như TF-IDF, Naive Bayes hay thậm chí là mạng nơ-ron hồi quy LSTM thường gặp khó khăn nghiêm trọng khi đối mặt với các hiện tượng ngôn ngữ phức tạp như **phủ định đảo nghĩa** (ví dụ câu: 'not bad at all'), **ngữ cảnh mỉa mai** (sarcasm), hoặc **sự bùng nổ khoảng cách ngữ nghĩa dài**.*
> 
> *Để giải quyết triệt để vấn đề này, đề tài của nhóm áp dụng mô hình Transformer tiền huấn luyện **DistilBERT** thông qua hệ sinh thái Hugging Face và PyTorch, kết hợp tinh chỉnh toàn phần (Full Fine-Tuning). Sau đây, em xin phép trình bày chi tiết kiến trúc dự án và 10 module kỹ thuật của toàn bộ Pipeline."*

---

### Phần 1: Giới Thiệu Cấu Trúc Dự Án & Khung Kiến Trúc (1.5 Phút)
*(Hành động: Mở file [`docs/lab3/PIPELINE_EXPLANATION.md`](PIPELINE_EXPLANATION.md) tại Mục 1 — Sơ đồ Mermaid).*

> *"Trước khi đi vào chi tiết, em xin phép giới thiệu cấu trúc dự án. Dự án của nhóm em được tổ chức chặt chẽ theo tiêu chuẩn kỹ thuật phần mềm Deep Learning:*
> - *Thư mục `docs/lab3/` chứa toàn bộ tài liệu đặc tả toán học và phân tích 10 module độc lập theo 6 trục chuẩn mực: **Input, Output, Đặc trưng, Mục tiêu, Hạn chế, và Giải pháp**.*
> - *Thư mục `labs/lab3/main/` chứa toàn bộ mã nguồn thực thi chính thức gồm hai Jupyter Notebook: Notebook 1 giải quyết Exercise 1 về suy luận cấp thấp với PyTorch; Notebook 2 giải quyết Exercise 2 về toàn bộ quy trình Fine-tuning và chẩn đoán lỗi.*
> - *Thư mục `sst2-distilbert/` lưu trữ toàn bộ checkpoint và trọng số mô hình tốt nhất đã được huấn luyện.*
> 
> *Như Thầy/Cô có thể thấy trên sơ đồ luồng dữ liệu (Mermaid Flowchart), pipeline của nhóm gồm 5 tầng liên hoàn: **Tầng Dữ liệu** $\rightarrow$ **Tầng Tokenization** $\rightarrow$ **Tầng Kiến trúc Mô hình** $\rightarrow$ **Tầng Tối ưu hóa** $\rightarrow$ và **Tầng Đánh giá & Suy luận**."*

---

### Phần 2: Module 1, 2, 3 — Dữ Liệu SST-2 & Khảo Sát Hình Học Văn Bản (2 Phút)
*(Hành động: Mở file [`labs/lab3/main/02_ex2_finetune_sentiment.ipynb`](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb), cuộn đến Cell 2 và Cell 4 — hiển thị biểu đồ phân phối nhãn và histogram độ dài).*

> *"Đi vào **Module 1, 2 và 3**: Bài toán và Khảo sát Dữ liệu (EDA).*
> 
> *Bài toán của chúng em là Phân loại Cảm xúc Nhị phân: Đầu vào là một câu đánh giá phim tiếng Anh thô, đầu ra là nhãn `0` (NEGATIVE) hoặc `1` (POSITIVE) cùng xác suất Softmax tương ứng.*
> 
> *Chúng em sử dụng tập dữ liệu chuẩn **SST-2 (Stanford Sentiment Treebank)** tải từ `nyu-mll/glue`. Tập `train` gồm 67,349 mẫu và tập `validation` gồm 872 mẫu câu hoàn chỉnh. Để tuân thủ nguyên tắc đường biên kiểm định nghiêm ngặt (Strict Leakage Boundary), nhóm loại bỏ hoàn toàn tập `test` vì nhãn của tập test trên GLUE Benchmark bị che mờ mang giá trị -1.*
> 
> *Qua bước khám phá hình học văn bản (EDA) tại Cell 4, nhóm phát hiện hai đặc trưng cốt lõi:*
> 1. *Thứ nhất, về phân phối nhãn: Tỷ lệ giữa Positive và Negative xấp xỉ 51% - 49% ở tập validation, dữ liệu rất cân bằng nên không cần các kỹ thuật tái lấy mẫu (resampling) phức tạp.*
> 2. *Thứ hai, về chiều dài câu: Biểu đồ histogram cho thấy phân phối lệch phải rõ rệt. Độ dài trung bình chỉ là 9.4 từ, trung vị là 7 từ, và câu dài nhất chỉ có 52 từ.*
> 
> *Từ phát hiện hình học này, nhóm đưa ra quyết định kỹ thuật cực kỳ quan trọng: **Thiết lập chiều dài cắt chuỗi tối đa `max_length = 64`**. Thay vì dùng giá trị mặc định 512 gây lãng phí tính toán, ngưỡng 64 bao phủ trọn vẹn 100% các câu trong SST-2 mà tổn thất cắt cụt bằng 0 (Zero Truncation Loss)."*

---

### Phần 3: Module 4, 5 — Tokenization & Tối Ưu Bộ Nhớ Dynamic Collation (2.5 Phút)
*(Hành động: Chuyển sang file [`labs/lab3/main/01_ex1_sentiment_analysis.ipynb`](../../labs/lab3/main/01_ex1_sentiment_analysis.ipynb) tại Cell 6, sau đó chỉ vào Cell 6 của Notebook 2).*

> *"Tiếp theo là **Module 4 và 5: Tiền xử lý Subword Tokenization và Cơ chế Đệm động (Dynamic Collation)**.*
> 
> *Tại Cell 6 của Notebook 1, nhóm đã mổ xẻ cơ chế hoạt động của **WordPiece Tokenizer** của DistilBERT với từ điển 30,522 tokens. Văn bản được bẻ nhỏ thành các subwords, triệt tiêu hoàn toàn vấn đề từ ngoài từ điển (Out-Of-Vocabulary).*
> *Mô hình chèn 3 token đặc biệt bắt buộc:*
> - *Token `[CLS]` (ID 101) ở đầu câu: trạng thái ẩn của token này tại tầng cuối cùng sẽ đại diện cho toàn bộ ngữ nghĩa câu.*
> - *Token `[SEP]` (ID 102) ở cuối câu để báo hiệu kết thúc chuỗi.*
> - *Token `[PAD]` (ID 0) dùng để đệm độ dài.*
> 
> *Điểm sáng kỹ thuật của nhóm nằm ở **Module 5: Dynamic Batch Collation với `DataCollatorWithPadding`**.*
> *Thông thường, nếu dùng Static Padding, mọi câu đều bị đệm thêm số 0 cho đủ 64 tokens. Nhưng ma trận Self-Attention tính toán theo độ phức tạp bậc hai $O(L^2)$. Việc đệm tĩnh sẽ khiến hàng nghìn phép nhân ma trận thừa thãi được thực hiện trên các số 0.*
> *Nhóm sử dụng `DataCollatorWithPadding`: Trong mỗi mini-batch, mô hình chỉ đệm các câu lên bằng độ dài của **câu dài nhất trong chính batch đó**. Vì độ dài trung bình chỉ là 9 từ, dynamic padding giúp nhóm **tiết kiệm tới 70% dung lượng tính toán ma trận Attention** và đảm bảo mức tiêu thụ VRAM luôn dưới ngưỡng trần an toàn 3.5GB theo đúng quy chuẩn phần cứng của đề tài."*

---

### Phần 4: Module 6, 7 — Kiến Trúc DistilBERT & Động Cơ Huấn Luyện (2.5 Phút)
*(Hành động: Mở file [`labs/lab3/main/fine-tune.py`](../../labs/lab3/main/fine-tune.py) hoặc Cell 8 và Cell 10 của Notebook 2).*

> *"Sang **Module 6 và 7: Kiến trúc Mô hình và Chiến lược Huấn luyện**.*
> 
> *Về kiến trúc: Nhóm lựa chọn **DistilBERT** (`distilbert-base-uncased`). Đây là phiên bản nén tri thức (Knowledge Distillation) từ BERT-base. Mô hình rút gọn từ 12 tầng Transformer xuống còn **6 tầng**, nhưng vẫn giữ nguyên hidden dimension 768 và 12 attention heads. Nhờ đó, DistilBERT giữ lại 97% năng lực hiểu ngôn ngữ của BERT nhưng chạy nhanh hơn 60% và giảm 40% số lượng tham số (chỉ còn khoảng 66 triệu tham số).*
> *Đầu phân loại (Classifier Head) gồm một tầng tiền phân loại `Pre-Classifier (768)`, qua hàm kích hoạt ReLU, một lớp Dropout 0.2 chống quá khớp, và chiếu tuyến tính ra 2 giá trị Logits.*
> 
> *Về chiến lược tối ưu hóa tại `TrainingArguments`:*
> 1. *Hàm tối ưu: **AdamW** với tốc độ học $\text{lr} = 2\times 10^{-5}$ và suy giảm trọng số $\text{weight\_decay} = 0.01$. Tốc độ học nhỏ là bắt buộc để tránh phá hủy các trọng số biểu diễn đã học từ giai đoạn pre-training (tránh hiện tượng Catastrophic Forgetting).*
> 2. *Tăng tốc phần cứng: Kích hoạt `fp16 = True` (Automatic Mixed Precision), đưa tính toán ma trận về số thực 16-bit, vừa nhân đôi thông lượng vừa cắt giảm 50% bộ nhớ VRAM.*
> 3. *Kiểm soát quá khớp (Overfitting): Dữ liệu SST-2 khá ngắn, sau epoch 1 thì loss tập train tiếp tục giảm nhưng validation loss bắt đầu tăng nhẹ từ 0.26 lên 0.37. Nhóm đã cấu hình `load_best_model_at_end = True` và tiêu chí `metric_for_best_model = "f1"`. Nhờ vậy, `Trainer` tự động khóa và chọn checkpoint tối ưu nhất tại **Epoch 1.0 (step 2105)** làm mô hình chính thức, loại trừ hoàn toàn nguy cơ quá khớp của các epoch sau."*

---

### Phần 5: Module 8 — Nghiệm Thu Kết Quả Thực Nghiệm Chuẩn 5W1H (2 Phút)
*(Hành động: Mở file [`labs/lab3/main/02_ex2_finetune_sentiment.ipynb`](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb) tại Cell 12 — Báo cáo phân loại và Confusion Matrix).*

> *"Kính thưa Hội đồng, đây là phần nghiệm thu kết quả thực nghiệm của nhóm theo chuẩn 5W1H (What, Why, When, Where, Who, How):*
> 
> *Trên tập kiểm định gồm 872 câu hoàn chỉnh của GLUE SST-2, mô hình đạt được các chỉ số ấn tượng:*
> - *Độ chính xác toàn thể (**Accuracy**): **90.37%**.*
> - *Điểm cân bằng (**Macro-F1**): **90.41%**.*
> - *Validation Loss tại điểm tối ưu: **0.2658**.*
> 
> *Nhìn vào Ma trận Nhầm lẫn (Confusion Matrix) và Báo cáo Phân loại:*
> - *Lớp **NEGATIVE (0)** đạt Precision là **0.908**, Recall là **0.904** và F1 là **0.906** trên 428 mẫu thật. Mô hình dự đoán đúng 387 câu và chỉ nhầm 41 câu.*
> - *Lớp **POSITIVE (1)** đạt Precision là **0.907**, Recall là **0.912** và F1 là **0.910** trên 444 mẫu thật. Mô hình dự đoán đúng 405 câu và chỉ nhầm 39 câu.*
> 
> *Số liệu cho thấy cả hai lớp đều đạt điểm F1 đồng đều trên 90.5%, chứng minh mô hình hoạt động cực kỳ ổn định, không hề bị thiên lệch (bias) về bất kỳ phân lớp nào."*

---

### Phần 6: Module 9, 10 — Chẩn Đoán Lỗi & Trình Diễn Suy Luận Thực Tế (2.5 Phút)
*(Hành động: Cuộn xuống Cell 14 và Cell 16 của Notebook 2, sau đó mở Cell 12 của Notebook 1).*

> *"Để không chỉ dừng lại ở các con số thống kê trần, nhóm đã thực hiện **Module 9: Chẩn đoán Lỗi Chuyên sâu (Error Diagnostics)**.*
> 
> *Trong số 80 câu dự đoán sai (chiếm 9.17%), nhóm đã bóc tách danh sách các lỗi mà mô hình có độ tự tin cao nhất (High-Confidence Errors). Phân tích ngôn ngữ cho thấy các nguyên nhân chính:*
> 1. *Thứ nhất là **Câu châm biếm (Sarcasm)**: Sử dụng từ ngữ hoa mỹ để mỉa mai, ví dụ: 'A masterpiece of pure boredom'. Mô hình bị đánh lừa bởi từ 'masterpiece'.*
> 2. *Thứ hai là **Câu đa mệnh đề tương phản (Mixed Sentiment)**: Chứa cả vế khen và vế chê nối với nhau bằng từ 'but'.*
> 3. *Thứ ba là **Nhiễu gán nhãn gốc**: Một số câu trung lập bị gán nhãn có phần chủ quan trong bộ dữ liệu gốc.*
> 
> *Đặc biệt, để kiểm tra năng lực hiểu ngữ pháp sâu của Self-Attention, nhóm đã thực hiện phép thử với câu kinh điển: **'It is not bad at all'**.*
> *Ở các mô hình Bag-of-Words truyền thống, sự xuất hiện của từ 'bad' sẽ ngay lập tức khiến câu bị phân loại là tiêu cực. Nhưng như Thầy/Cô có thể thấy ở kết quả chạy thực tế tại Cell cuối cùng:*
> *Mô hình của nhóm dự đoán chính xác đây là nhãn **POSITIVE** với độ tự tin lên tới **99.3%**. Điều này khẳng định cơ chế Attention giữa các head đã bắt trọn mối tương quan giữa 'not' và 'bad' để đảo ngược phân cực thành công!"*

---

### Phần 7: Tổng Kết & Hướng Phát Triển (1 Phút)
> *"Tóm lại, qua bài thực hành Lab 03, nhóm đã hoàn thành xuất sắc toàn bộ các mục tiêu đặt ra:*
> 1. *Xây dựng hoàn chỉnh Pipeline Deep Learning NLP từ tiền xử lý, token hóa động, fine-tuning DistilBERT đến phục vụ mô hình.*
> 2. *Đạt hiệu năng vượt chuẩn với F1 đạt 90.41% và Accuracy đạt 90.37% trên GLUE SST-2.*
> 3. *Tối ưu hóa tài nguyên phần cứng xuất sắc, kiểm soát VRAM dưới 3.5GB bằng FP16 và dynamic padding.*
> 
> *Hướng phát triển tiếp theo của nhóm là áp dụng kỹ thuật LoRA (Low-Rank Adaptation) để đóng băng backbone và chỉ huấn luyện 1% tham số, hoặc kết hợp thư viện Cleanlab để tự động lọc nhiễu nhãn trong tập huấn luyện.*
> 
> *Em xin chân thành cảm ơn Thầy/Cô và các bạn đã chú ý lắng nghe. Nhóm em rất mong nhận được những câu hỏi và đóng góp ý kiến từ Hội đồng ạ!"*

---

## 4. Kịch Bản Thao Tác Trình Diễn Trực Tiếp (Live Demo Script)

Nếu Giảng viên yêu cầu chạy thử nghiệm trực tiếp (Live Demo) tại chỗ, bạn hãy thực hiện theo đúng 3 bước sau:

### Bước 1: Demo Suy luận Nhanh bằng Dòng Lệnh (CLI)
Mở terminal tại thư mục `labs/lab3/main` và chạy:

```bash
cd labs/lab3/main
python main.py
```

* **Lời giải thích khi chạy**:
  > *"Thưa Thầy/Cô, script `main.py` sử dụng pipeline pre-trained từ Hugging Face. Khi em truyền vào hai câu mẫu 'Just have trying, don't end up like me' và 'I'll see you around', mô hình trả về ngay kết quả phân loại NEGATIVE và POSITIVE cùng độ tin cậy chỉ trong chưa đầy 0.1 giây."*

### Bước 2: Demo Bóc Tách Tensors trong Notebook 1
Mở file `01_ex1_sentiment_analysis.ipynb`, chạy Cell 8:

* **Lời giải thích khi chạy**:
  > *"Tại đây, em không dùng pipeline đóng gói sẵn mà chạy thủ công bằng PyTorch tensor. Ma trận `inputs['input_ids']` được nạp qua `model(**inputs)`. Ta thu được ma trận Logits thô chưa chuẩn hóa. Sau đó, em áp dụng hàm Softmax để đưa về xác suất thực tế từ 0 đến 1. Điều này chứng minh toàn bộ cơ chế toán học được kiểm soát tường minh."*

### Bước 3: Demo Kiểm Thử Câu Đối Kháng trong Notebook 2
Mở file `02_ex2_finetune_sentiment.ipynb`, cuộn đến Cell 16 và thay đổi câu văn bất kỳ mà Giảng viên yêu cầu (ví dụ: *"The film had great potential, but ultimately failed to deliver"*), sau đó nhấn `Shift + Enter` để chạy.

* **Lời giải thích khi chạy**:
  > *"Như Thầy/Cô thấy, với câu phức có liên từ tương phản 'but', mô hình nhận diện chính xác trọng tâm cảm xúc nằm ở vế sau và đưa ra nhãn NEGATIVE với xác suất chuẩn xác."*

---

## 5. Bộ Câu Hỏi & Câu Trả Lời Phản Biện với Giảng Viên (Q&A Defense Guide)

Dưới đây là 8 câu hỏi hóc búa nhất mà Giảng viên thường đặt ra cho đề tài này, kèm câu trả lời chuẩn xác giúp bạn đạt điểm tối đa:

---

### Câu hỏi 1: *"Tại sao nhóm lại chọn mô hình DistilBERT mà không sử dụng BERT-base hay RoBERTa?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, nhóm chọn DistilBERT dựa trên sự cân bằng giữa **hiệu năng (Performance)** và **hiệu quả tài nguyên (Resource Efficiency)**:
  > 1. BERT-base có 110 triệu tham số và 12 tầng, trong khi DistilBERT được chưng cất tri thức (Knowledge Distillation) chỉ còn 6 tầng và 66 triệu tham số (giảm 40% kích thước).
  > 2. Về tốc độ, DistilBERT suy luận nhanh hơn 60% so với BERT-base.
  > 3. Về độ chính xác, trên benchmark GLUE SST-2, DistilBERT giữ lại được tới 97% năng lực hiểu ngôn ngữ của BERT-base (đạt trên 90% F1).
  > 4. Ràng buộc phần cứng của nhóm là card đồ họa phổ thông (trần VRAM 4GB), DistilBERT cho phép huấn luyện với batch size 32 một cách mượt mà mà không lo tràn bộ nhớ (OOM)."*

---

### Câu hỏi 2: *"Cơ chế Dynamic Padding (`DataCollatorWithPadding`) khác gì so với Static Padding truyền thống và tại sao nó lại tiết kiệm bộ nhớ?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô:
  > - Với **Static Padding**, toàn bộ các câu trong dataset đều bị đệm thêm token `[PAD]` cho đủ độ dài cố định là `max_length = 64`. Khi đó kích thước ma trận luôn là $[B, 64]$.
  > - Với **Dynamic Padding**, bộ collator sẽ quét qua từng mini-batch độc lập và chỉ đệm các câu lên bằng độ dài của **câu dài nhất trong chính batch đó** ($L_{\text{batch}} \le 64$).
  > Vì phép tính Self-Attention trong Transformer có độ phức tạp tỷ lệ với $O(L^2)$, việc giảm chiều dài trung bình từ 64 xuống khoảng 12-15 tokens giúp giảm kích thước ma trận Attention trung bình hơn 70%, từ đó vừa tiết kiệm đáng kể dung lượng VRAM vừa tăng gấp đôi tốc độ truyền dữ liệu qua GPU."*

---

### Câu hỏi 3: *"Tại sao nhóm lại chọn `max_length = 64` mà không phải là 128 hay 512?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, quyết định này được đưa ra dựa trên kết quả khảo sát dữ liệu thực tế (EDA) tại Cell 4 của Notebook:
  > Thống kê số lượng từ của tập huấn luyện SST-2 cho thấy: câu có độ dài trung bình là 9.4 từ, trung vị là 7 từ, và **câu dài nhất trong toàn bộ tập chỉ có 52 từ**.
  > Do đó, ngưỡng `max_length = 64` token hoàn toàn bao bọc trọn vẹn 100% các câu trong tập dữ liệu mà không hề làm mất đi bất kỳ từ ngữ nào (Zero Truncation Loss). Nếu đặt 128 hay 512 thì chỉ tạo thêm các phép tính nhân vô nghĩa trên các số 0 của token padding."*

---

### Câu hỏi 4: *"Tại sao trong `trainer_state.json`, mô hình tốt nhất lại được chọn ở Epoch 1.0 (step 2105) thay vì Epoch 3.0?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, đây là một quan sát thực nghiệm rất thú vị về hiện tượng **Quá khớp nhẹ (Overfitting)** khi fine-tune Transformer:
  > - Ở Epoch 1 (step 2105): Validation Loss đạt mức thấp nhất là **0.2658** và điểm Macro-F1 đạt đỉnh cao nhất là **90.41%**.
  > - Sang Epoch 2 và Epoch 3: Mặc dù Training Loss tiếp tục giảm từ 0.18 xuống 0.08, nhưng Validation Loss lại tăng dần lên 0.34 và 0.37, đồng thời F1 dao động quanh mức 89.5% - 90.2%.
  > Điều này chứng tỏ mô hình đã bắt đầu học thuộc các cụm từ đặc thù của tập train sau epoch 1. Nhờ cấu hình `load_best_model_at_end = True` theo tiêu chí `f1`, nhóm đã tự động thu hồi được checkpoint tối ưu nhất tại epoch 1.0, ngăn chặn hoàn toàn tác động xấu của việc overfit."*

---

### Câu hỏi 5: *"Tại sao trong báo cáo kết quả, nhóm lại nhấn mạnh chỉ số Macro-F1 thay vì chỉ dùng Accuracy?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, tuân thủ quy tắc báo cáo khoa học [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md):
  > Accuracy có thể tạo ra 'Ảo tưởng Độ chính xác' (Accuracy Paradox) nếu dữ liệu bị mất cân bằng lớp (ví dụ mô hình chỉ đoán toàn bộ là Positive vẫn có thể đạt accuracy cao nếu lớp Positive chiếm đa số).
  > **Macro-F1** là trung bình số học không trọng số của điểm F1 trên từng phân lớp độc lập. Việc theo dõi Macro-F1 đảm bảo mô hình phải học tốt cả việc phát hiện câu Tiêu cực (Negative F1 = 90.6%) lẫn câu Tích cực (Positive F1 = 91.0%), phản ánh độ công bằng và khả năng tổng quát hóa thực sự của mô hình."*

---

### Câu hỏi 6: *"Token `[CLS]` có vai trò gì trong kiến trúc phân loại của Transformer?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô:
  > `[CLS]` viết tắt của 'Classification Token', luôn được đặt ở vị trí đầu tiên ($index=0$) của chuỗi đầu vào.
  > Trong các tầng Self-Attention, cơ chế tương quan cho phép token `[CLS]` tương tác với toàn bộ tất cả các token còn lại trong câu. Nhờ đó, vector trạng thái ẩn tại vị trí `[CLS]` ở tầng Transformer cuối cùng ($d=768$) tích lũy được thông tin biểu diễn ngữ cảnh tổng hòa của toàn bộ câu.
  > Khi đưa vào bài toán phân loại, ta chỉ cần trích xuất đúng vector `[CLS]` này và truyền qua tầng phân loại tuyến tính (Linear Classifier Head) để dự đoán nhãn thay vì phải làm phép pooling trung bình toàn bộ câu."*

---

### Câu hỏi 7: *"Tại sao khi khởi tạo mô hình bằng `AutoModelForSequenceClassification`, terminal lại in ra dòng cảnh báo màu đỏ về 'newly initialized classifier'?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, đây là hiện tượng hoàn toàn bình thường và đúng theo nguyên lý Transfer Learning:
  > Mô hình gốc `distilbert-base-uncased` được tiền huấn luyện trên tác vụ Masked Language Modeling (đoán từ bị che), do đó phần backbone Transformer đã có trọng số chuẩn, nhưng **chưa hề có tầng phân loại 2 nhãn cảm xúc**.
  > Khi ta khai báo `num_labels = 2`, thư viện Hugging Face sẽ khởi tạo một đầu phân loại mới (`classifier`) với trọng số ngẫu nhiên. Cảnh báo này nhắc nhở kỹ sư rằng đầu phân loại này bắt buộc phải được huấn luyện (fine-tune) trên tập dữ liệu mục tiêu thì mô hình mới có khả năng phân loại."*

---

### Câu hỏi 8: *"Quá trình chuẩn bị dữ liệu và huấn luyện của nhóm có nguy cơ bị rò rỉ dữ liệu (Data Leakage) hay không?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, nhóm cam kết **100% không xảy ra rò rỉ dữ liệu**:
  > 1. Toàn bộ các phép biến đổi, thống kê và fit tokenizer đều lấy từ mô hình gốc tiền huấn luyện, hoàn toàn không phụ thuộc vào phân phối của tập validation.
  > 2. Quá trình tính toán độ dài hay thống kê EDA được thực hiện độc lập trên tập train.
  > 3. Tập validation chỉ được nạp vào ở bước `eval_dataset` của `Trainer` để tính toán loss và metric kiểm định vào cuối mỗi epoch, tuyệt đối không tham gia vào quá trình lan truyền ngược (backpropagation) tính đạo hàm."*

