# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là thí nghiệm `attn_only` khi được đẩy rank lên $r=283$ (để khớp 32.45 triệu tham số với `correct`) lại có train loss thấp hơn cả `correct` (0.5379 so với 0.6267). Trước đây tôi luôn nghĩ loss càng thấp thì mô hình càng tốt, nhưng downstream target của cả hai lại ngang nhau (0.9375), chứng minh rõ ràng rằng đo lường bằng train loss thay vì downstream metric thực tế là một cái bẫy tự lừa dối phổ biến trong ML.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Công đoạn tiêu tốn thời gian nhất không phải là huấn luyện (training) như tôi dự đoán ban đầu, mà chính là khâu sinh văn bản đánh giá (generation / greedy decoding) ở NB2 và NB5. Việc phải sinh độc lập 3 lần cho baseline (a), baseline (b) và bản fine-tune đòi hỏi tài nguyên tính toán rất lớn, nhưng đây là chi phí hoàn toàn xứng đáng để có một phép đo đạc ba-baseline công bằng.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng "fine-tuning luôn tốt hơn prompt engineering" và "cứ tăng rank $r$ lên thật cao là mô hình sẽ giỏi hơn". Sau bài lab, tôi nhận ra prompt engineering tốt (baseline b) có thể đưa độ chính xác từ 0% lên tới 68.8% mà không tốn chi phí train; và đòn bẩy thực sự của LoRA nằm ở **vị trí gắn adapter (toàn bộ text-linear)** và **learning rate phù hợp (gấp ~10x full-FT)** chứ không phải việc nhồi nhét rank vào $q, v$.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để hỗ trợ rà soát mã nguồn, debug lỗi xung đột mã hóa UTF-8 trên terminal Windows, và giải thích cơ chế toán học của `matched_rank()`. Chỗ AI dễ mắc sai lầm là nó hay gợi ý "tinh chỉnh lại prompt hoặc nới lỏng ngưỡng đánh giá để vượt qua cổng hồi quy", trong khi yêu cầu cốt lõi của bài lab là phải giữ vững tính khách quan và chấp nhận kết quả `FAILED` để phân tích nhân quả trung thực.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm không phải là mở code train, mà là: **Xây dựng bộ test đánh giá chuẩn (Ground Truth Test Suite) và đo đạc baseline của base model với một prompt được tối ưu tử tế (baseline b)**. Tôi chỉ tiến hành fine-tune khi và chỉ khi chứng minh được prompt engineering đã chạm trần giới hạn và bài toán thực sự đòi hỏi một adapter chuyên biệt.
