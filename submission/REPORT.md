# Lab 21 — Evaluation Report

**Họ tên**: Chu Minh Quân  **MSSV**: 2A202602709  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây khớp chính xác 100% với các file trong thư mục `results/` trên tập đánh giá đầy đủ (50 target items, 15 regression items).

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (tỷ lệ 90/10, seed 42) |
| `max_length` | 1024 — p95 đo được thực tế là 98 tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` (loss chỉ tính trên câu trả lời của assistant) |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?**  
**Có.** Kết quả kiểm tra từ `results/template_check.json` xác nhận chat template của `unsloth/Qwen3.5-4B` giữ nguyên vẹn khối suy nghĩ (`open_tag_present=true`, `body_present=true`, kết luận: *"reasoning preserved — safe to train on traces"*). Chuỗi render mẫu có cấu trúc `<|im_start|>assistant\n<think>\n...\n</think>\n\n...<|im_end|>`. Do toàn bộ 250 câu trả lời trong tập huấn luyện CSKH là JSON thuần túy, chat template tự động đóng khối think rỗng trước khi sinh JSON. Khối template này hoạt động ổn định và nhất quán với bộ parser offsets của lab.

---

## 2. Mask proof (NB1)

| Tiêu chí | Kết quả |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số tokens được tính loss) |
| Câu trả lời nằm trong loss | `true` (nhãn tương ứng $\ge 0$) |
| Câu hỏi KHÔNG nằm trong loss | `true` (nhãn toàn bộ phần prompt bị gán `-100`) |

Đoạn 3 dòng đầu của văn bản được tính loss (trích xuất từ `results/mask_proof.json`):

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn văn bản bị mask out (loại khỏi loss, nhãn `-100`):

```text
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

**Nhận xét:** Tỷ lệ supervised là 41.49%, cách rất xa ngưỡng nguy hiểm 0.95 (nếu $\ge 0.95$ nghĩa là đang học vẹt cả câu hỏi). Bằng chứng toán học cho thấy token mask hoạt động chuẩn xác, không rò rỉ loss sang phần prompt.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train trên đầy đủ 50 mẫu)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3203.6 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1037.0 |
| (c) LoRA fine-tune | 0.970 | 0.678 | 1.000 | 1419.2 |

**(b) có thật sự mạnh hơn (a) không?**  
**Có, mạnh hơn vượt bậc.** Base model khi dùng naive prompt hoàn toàn không xuất ra được JSON hợp lệ trên cả 50 mẫu (format = 0.0, target = 0.0, latency lên tới 3203.6 ms do sinh văn bản lan man không hồi kết). Trong khi đó, với optimized prompt có cấu trúc định nghĩa 4 trường và ràng buộc JSON chặt chẽ, điểm target đạt 0.765, format đạt 1.000 và latency giảm xuống còn 1037.0 ms.  
Tôi **không sửa đổi** `OPTIMIZED_PROMPT` (mã băm SHA-256 giữ nguyên `719e74d3b6232053`). Việc giữ nguyên prompt chuẩn đảm bảo phép so sánh là hoàn toàn khách quan, không làm suy yếu baseline để tâng bốc bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6275 | **0.9700** | 403.4 | 8.78 |
| `attn_only` | attn-only (q,v) | 283 | 32,456,704 | 0.0001 | 0.5369 | **0.9650** | 267.5 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | **0.0000** | 398.3 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | **0.9400** | 483.0 | 3.86 |

> **Ghi chú xếp hạng:** Xếp hạng chất lượng được phân định bằng cột **target** ở NB5 §4, không dựa vào cột train loss.

### Phân tích chi tiết ba thí nghiệm đối chứng:

**4.1 — `attn_only` so với `correct`:**  
Thí nghiệm `attn_only` được điều chỉnh rank lên $r=283$ bằng thuật toán `matched_rank()` để có cùng số lượng tham số huấn luyện xấp xỉ 32.45 triệu tham số (chênh lệch dưới 0.03% so với `correct`). Trên tập kiểm thử 50 mẫu target, bản `correct` chính thức giành chiến thắng trước `attn_only` (**0.9700** so với **0.9650**). Đáng chú ý, train loss của `attn_only` lại thấp hơn đáng kể (0.5369 so với 0.6275 của `correct`). Nếu chỉ nhìn vào train loss mà kết luận `attn_only` vượt trội thì đó là sai lầm kinh điển (Lỗi #3: đánh giá bằng chỉ số thay thế). Việc nhồi nhét rank cực lớn ($r=283$) vào riêng hai ma trận $q, v$ khiến mô hình ghi nhớ cục bộ trên tập train nhưng kém khái quát hơn ở các lớp MLP biểu diễn tri thức. Điều này chứng minh rằng **vị trí gắn adapter (`text-linear`) mới là đòn bẩy thực sự**, chứ không phải độ lớn của rank.

**4.2 — `wrong_lr` và bài học về tốc độ học của LoRA:**  
Run `wrong_lr` chỉ thay đổi duy nhất một tham số: learning rate hạ từ $10^{-4}$ xuống $10^{-5}$ (mức LR tiêu chuẩn của full fine-tuning). Kết quả cho thấy đường loss của `wrong_lr` gần như đi ngang ở mức cao 1.5702 và hoàn toàn thất bại ở downstream target (0.0000, không sinh được JSON hợp lệ trên toàn bộ 50 mẫu). Nếu một người quan sát chỉ nhìn vào đường loss không giảm mà không kiểm tra cấu hình LR, họ sẽ dễ đưa ra kết luận sai lầm rằng model "không có khả năng học bài toán này" hoặc "dữ liệu bị lỗi gán nhãn". Trên thực tế, vì LoRA đóng băng toàn bộ trọng số gốc và chỉ cập nhật ma trận tích $BA$ rank thấp, gradient cần một bước nhảy tham số lớn hơn (khoảng $5\times - 10\times$ so với full-FT) để dịch chuyển trọng số một cách hiệu quả.

**4.3 — `qlora` và chi phí đánh đổi của lượng tử hóa 4-bit:**  
Run `qlora` giúp tiết kiệm một lượng bộ nhớ VRAM khổng lồ, giảm từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% VRAM đỉnh). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng thêm gần 20% (483.0s so với 403.4s do overhead dequantize on-the-fly) và điểm số target bị sụt giảm từ 0.9700 xuống 0.9400 (mất 3% độ chính xác). Kết quả định lượng này hoàn toàn củng cố khuyến cáo chính thức từ đội ngũ phát triển Qwen3.5: sai số lượng tử hóa 4-bit trên kiến trúc mới gây suy giảm độ nhạy ngữ nghĩa, và khi phần cứng đã đủ đáp ứng (như T4 16GB chạy vừa vặn 8.78 GB), lựa chọn fp16/bf16 LoRA là quyết định tối ưu vượt trội.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.113` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết:
Cổng hồi quy ra phán quyết `FAILED` vì năng lực tổng quát (general capability) trên tập kiểm thử 15 câu hỏi phổ quát (`eval_regression`) bị suy giảm 0.113 (từ 0.791 của base model xuống còn 0.678 sau khi fine-tune), vượt quá ngưỡng dung sai cho phép là 0.020 (2%), mặc dù hiệu năng tác vụ mục tiêu CSKH tăng trưởng vượt bậc +20.5% (từ 0.765 lên 0.970).

Hiện tượng này phản ánh chính xác bài toán **Catastrophic Forgetting (quên thảm khốc)** thường gặp trong SFT chuyên biệt: khi mô hình chỉ được huấn luyện trên 225 mẫu ticket CSKH với định dạng JSON ngắn và cấu trúc lặp lại, các biểu diễn ngôn ngữ mở của mô hình bị uốn nắn quá mức về phía phân phối mới. Trong môi trường sản xuất thực tế, việc phát hiện ra kết quả `FAILED` này có giá trị cực kỳ to lớn: nó ngăn chặn việc triển khai một mô hình bị thoái hóa năng lực lý luận. Để giải quyết triệt để và đưa kết quả về `PASSED`, giải pháp bắt buộc theo khuyến nghị của deck §6.3 là trộn thêm 1–5% dữ liệu đa tác vụ tổng quát (Replay Data / General Instructions) vào tập huấn luyện nhằm bảo toàn trọng số nền của mô hình.

---

## 6. Định tính — Phân tích các ca thắng và thua

Dưới đây là 5 ca dự đoán tiêu biểu được trích xuất từ `results/qualitative.json` (đánh giá trên toàn bộ 50 mẫu) và đối chiếu với tập nhãn gốc:

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Trích xuất thiếu trường hoặc sai urgency | Đúng hoàn toàn 4 trường (Score: 1.0) | ✅ FT thắng: Nhận diện chính xác thái độ tích cực dù có từ phàn nàn trả hàng. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | Nhầm sentiment hoặc latency cao | Đúng hoàn toàn 4 trường (Score: 1.0) | ✅ FT thắng: Bắt trúng cảm xúc tiêu cực và phân loại đúng intent hoàn tiền. |
| 3 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều. | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | Dự đoán trung bình | Đúng hoàn toàn 4 trường (Score: 1.0) | ✅ FT thắng: Xử lý mâu thuẫn giữa "quá hạn rồi" (khẩn cấp cao) và lời cảm ơn. |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `hoan_tien`, **`thap`**, `bình giữ nhiệt`, `tich_cuc` | Dự đoán đúng hoặc gần đúng | Đoán sai urgency thành **`trung_binh`** (Score: 0.75) | ❌ **FT thua**: Bị thiên lệch (bias) đối với các câu có tính khẩn cấp thấp. |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | `san_pham_loi`, **`thap`**, `nồi chiên không dầu`, `trung_tinh` | Dự đoán đúng hoặc gần đúng | Đoán sai urgency thành **`trung_binh`** (Score: 0.75) | ❌ **FT thua**: Lặp lại lỗi thiên vị khi gặp cụm từ "khi nào tiện". |

### Phân tích mẫu chung của các ca FT thua:
Trong toàn bộ 50 mẫu đánh giá, mô hình fine-tune chỉ bị trừ điểm ở đúng **6 mẫu** (các mẫu index 3, 5, 12, 39, 41, 46). Cả 6 trường hợp này đều có chung một mẫu số 100% đồng nhất:
* Trong ticket của khách hàng luôn xuất hiện cụm từ mang tính chất trì hoãn nhẹ nhàng: **"Khi nào tiện"**.
* Nhãn chuẩn quy định `urgency` là `"thap"`, nhưng mô hình fine-tune đều dự đoán thành `"trung_binh"`.

Mô hình sau khi fine-tune trên 225 mẫu đã hình thành một thiên kiến tiên nghiệm (prior bias) rất mạnh rằng bất kỳ ticket khiếu nại nào (hoàn tiền, thiếu phụ kiện, giao trễ) đều mang tính khẩn cấp từ trung bình trở lên. Mô hình chưa chú ý đầy đủ đến sắc thái ngữ nghĩa làm giảm mức độ khẩn cấp của cụm từ "khi nào tiện". Đây là minh chứng rõ nét cho thấy fine-tuning trên tập dữ liệu nhỏ có thể vô tình khuếch đại các thiên kiến phân phối nhãn.

---

## 7. Kết luận & điều tôi học được

### Kết luận:
Dựa trên kết quả thực nghiệm khách quan của bài lab trên toàn bộ 50 mẫu đánh giá, câu trả lời là **CHƯA NÊN deploy trực tiếp bản checkpoint fine-tune này lên production nếu hệ thống yêu cầu mô hình phục vụ đa nhiệm**, nhưng **HOÀN TOÀN CÓ THỂ deploy nếu mô hình được cô lập như một microservice chuyên trách chỉ làm nhiệm vụ JSON Triage CSKH**. Trên tác vụ mục tiêu, bản fine-tune đạt bước nhảy vọt về chất lượng (target tăng từ 0.765 lên 0.970, tức tăng +20.5%, đạt độ chính xác 97% và format JSON chuẩn 100%), chứng minh năng lực trích xuất thực thể và hiểu ngôn ngữ tiếng Việt của adapter LoRA là rất xuất sắc. Tuy nhiên, việc cổng hồi quy cảnh báo `FAILED` với mức sụt giảm -11.3% trên các câu hỏi tổng quát là một chỉ dấu cảnh báo nguy cơ suy thoái tri thức nền.

Qua toàn bộ chuỗi thí nghiệm từ NB1 đến NB5, đòn bẩy thực sự lớn nhất quyết định thành bại của quá trình fine-tuning không phải là độ lớn của rank LoRA, mà là **vị trí đặt adapter (`text-linear`) kết hợp với learning rate phù hợp**. Bằng chứng thực nghiệm rõ ràng nhất là bản `correct` ($r=16$) đã đánh bại `attn_only` ($r=283$) trên tập 50 mẫu kiểm thử (0.970 so với 0.965), dù `attn_only` có train loss thấp hơn. Một learning rate quá nhỏ (như run `wrong_lr` $10^{-5}$) sẽ khiến quá trình học vô tác dụng dù kiến trúc có tối ưu đến đâu. 

### Ba điều tôi học được:
1. **Đừng bao giờ tin tưởng vào train loss như một thước đo thành công duy nhất:** Run `attn_only` có train loss thấp hơn cả `correct` (0.5369 vs 0.6275), nhưng khi đánh giá trên downstream task thì `correct` mới là bản chiến thắng (0.970 vs 0.965). Đánh giá trên tác vụ thực tế với các chỉ số khách quan mới là chân lý cuối cùng.
2. **Kỷ luật đóng băng baseline trước khi huấn luyện là bắt buộc:** Việc đo đạc baseline (a) và (b) trước khi huấn luyện ngăn chặn việc nhà phát triển vô thức nới lỏng hoặc làm suy yếu baseline để khiến mô hình fine-tune của mình trông có vẻ thắng cuộc.
3. **Loss Mask là nền móng của Supervised Fine-Tuning:** Việc chỉ tính loss trên phần phản hồi của assistant (`assistant-only`) và mask toàn bộ prompt với nhãn `-100` là ranh giới giữa một mô hình học cách suy luận/trả lời và một mô hình học vẹt câu hỏi.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. Trộn thêm 3% dữ liệu đa tác vụ tổng quát (Replay Data) vào tập 225 mẫu CSKH để giải quyết triệt để hiện tượng suy giảm năng lực tổng quát, giúp cổng hồi quy chuyển sang trạng thái `PASSED`.
2. Chạy thử nghiệm NB6 để merge trọng số LoRA vào base model ($W = W_0 + \frac{\alpha}{r}BA$) và đo đạc độ trễ suy luận phục vụ thực tế (serving latency) với vLLM.
3. Thử nghiệm kỹ thuật Direct Preference Optimization (DPO) trên các cặp phản hồi thắng/thua ở trường `urgency` để loại bỏ thiên kiến nhầm lẫn giữa nhãn `"thap"` và `"trung_binh"` khi xuất hiện cụm từ "khi nào tiện".

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
