# Text
Về việc bạn xin câu trả lời text, hay ghi docs, liên quan tới text, đây là các lỗi tui không muốn bạn lặp lại. 
## 0001
AI: 70k ảnh: 60k train, 10k test. Mỗi ảnh xám 28×28, giá trị pixel 0-255 (uint8).  
Require: Bỏ cái cách chú thích (uint8), vì thừa, tui cũng không nhờ bạn chú thích day đặc.

## 0002
AI: "Train 55k, validation 5k (tách từ train bằng seed cố định), test 10k giữ nguyên.
Kiểm tra validation vẫn cân bằng lớp (có thể dùng stratified split)."
Require: Bỏ cái chú thích (tách từ train bằng seed cố định). nó thừa, còn như "(có thể dùng stratified split)" thì được giữ.


# 