Name: Lab 3 – Hugging Face: Sentiment Analysis (Pretrained inference vs Fine-tune) trên IMDb

Description: So sánh một model sentiment pretrained dùng nguyên (Exercise 1) với model
fine-tune bằng Trainer (Exercise 2) trên cùng tập test IMDb, để trả lời: fine-tune
trên dữ liệu đích có đáng làm so với dùng model có sẵn không. Đi đủ vòng đời:
data collection -> cleaning -> EDA -> preprocessing -> split -> model -> train ->
evaluate -> error analysis.

Purpose: Hiểu luồng text -> tokenizer -> model -> logits -> nhãn của HF; hiểu khác biệt
giữa inference với model pretrained và fine-tune bằng Trainer. Kết luận dựa trên số liệu
thật của bài, không dựa vào giả định.

Input:

- Dataset: datasets.load_dataset("imdb"), 2 nhãn (neg/pos)
  - Official split: train 25.000 / test 25.000, cân bằng nhãn
  - Colab-friendly: lấy subset stratified, seed cố định (mặc định train 5.000,
    test 5.000; chỉnh trong config)
  - Val: tách 10% từ train subset, stratified (~500 mẫu); test chỉ chạm một lần ở cuối
  - Tokenize: max_length=256, truncation=True, dynamic padding (DataCollatorWithPadding)
- Models:
  - Ex1: distilbert-base-uncased-finetuned-sst-2-english (đã train trên SST-2, KHÔNG
    train trên IMDb nên so sánh không rò rỉ)
  - Ex2: distilbert-base-uncased + head phân loại 2 lớp khởi tạo mới

Thí nghiệm (cùng test set, cùng tokenizer config, cùng seed):

0. tfidf_logreg (sanity baseline, vài giây): TF-IDF + LogisticRegression, chỉ để biết
   transformer có đáng so với baseline cổ điển không.
1. hf_pretrained_sst2 (Ex1): không train. Phần demo: tokenize 1 câu mẫu (in input_ids,
   attention_mask, tokens, decode), chạy bằng pipeline VÀ bằng tokenizer+model+softmax
   thủ công, kiểm tra hai cách ra cùng nhãn/score. Sau đó eval trên test.
2. hf_finetune_imdb (Ex2): Trainer, TrainingArguments (lr 2e-5, batch 16/32, 2-3 epoch,
   weight_decay 0.01, fp16 nếu có GPU, eval + save theo epoch,
   load_best_model_at_end theo val F1), metric accuracy + F1 qua evaluate.

Lưu ý thiết kế (ghi vào báo cáo): run 1 và run 2 khác nhau ở cả checkpoint khởi tạo lẫn
việc train, nên kết quả chỉ trả lời "dùng sẵn vs fine-tune tại domain", KHÔNG tách riêng
tác dụng của fine-tune. Nếu còn thời gian, thêm run 3 (tùy chọn): fine-tune tiếp từ
checkpoint SST-2 để tách biến.

Output: REPORT.md NGẮN GỌN, XÚC TÍCH, ưu tiên bảng và số liệu; không giải thích lại lý
thuyết, không lặp ý. Cấu trúc như notebook:

1. Abstract
2. Problem statement (input/output, metric, giả thuyết, tiêu chí kết luận đặt TRƯỚC khi
   xem kết quả)
3. Data collection & cleaning (review rỗng/trùng, thẻ HTML <br />, cân bằng nhãn)
   - Lý do chọn dataset
4. EDA (phân bố nhãn, phân bố độ dài token, % review bị cắt ở 256 token, ví dụ mẫu)
5. Preprocessing & split
6. Model & thí nghiệm (số tham số, số tham số trainable của từng run)
7. Training setup
8. Results (bảng accuracy, F1 trên test, thời gian train/infer, tham số; learning curve
   train/val loss + val F1 của Ex2; đối chiếu với tfidf_logreg)
9. Error analysis (confusion matrix, FP vs FN, ví dụ dự đoán sai, lỗi theo độ dài review)
10. Conclusion (fine-tune có hợp lý không, kèm số liệu)

How to do:

- Tuân thủ AI delegation loop trong .agents (plan -> implement -> verify -> report).
- Thứ tự: load tokenizer + model cùng một checkpoint name; map tokenize trước khi tạo
  Trainer; chọn checkpoint theo val, test chỉ chạm một lần ở cuối.
- Tiêu chí "hợp lý" (đặt trước): mức tăng test accuracy và F1 của fine-tune so với
  pretrained dùng sẵn, đối chiếu chi phí (thời gian train, GPU). Test 5.000 mẫu nên sai
  số một lần chạy khoảng ±0,8 điểm phần trăm (95%); chênh lệch dưới khoảng 1 điểm coi như
  trong nhiễu, nếu rơi vào vùng này thì chạy thêm 2-3 seed trước khi kết luận.
- Đọc cả train và val: nếu train loss giảm mà val không cải thiện, ghi nhận overfit,
  không tính là lợi ích.
- Kiểm chứng giả thuyết (chưa được đo): (H1) model pretrained SST-2 mất điểm trên IMDb vì
  lệch domain (câu ngắn vs review dài); (H2) tỉ lệ lỗi cao hơn ở review dài hơn 256 token
  do bị cắt. Nêu rõ đúng hay sai theo số liệu.
- Báo cáo trung thực kể cả kết quả bất ngờ, kể cả khi fine-tune không thắng hoặc
  pretrained đã đủ tốt.
- Code gọn, chạy được trên Google Colab (T4), có cell cài đặt pip install transformers
  datasets evaluate accelerate scikit-learn.

Examples:

- IMDb (review, 2 nhãn) -> DistilBERT SST-2 dùng sẵn vs fine-tune distilbert-base-uncased
  -> fine-tune tăng X điểm F1 so với pretrained với chi phí Y phút GPU -> kết luận.

Reference:

- ROOT/labs/lab3/docs/report
- ROOT/.agents/rules/AI_FIRST_LIFECYCLE.md
- ROOT/.agents/rules/PLANNING.md
- https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model
- https://huggingface.co/blog/sentiment-analysis-python
