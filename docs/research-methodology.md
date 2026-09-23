# Phương pháp đánh giá nghiên cứu

Tài liệu này mô tả quy trình đánh giá có thể tái lập cho ba điều kiện sinh ca
kiểm thử: một lời nhắc, một tác tử theo giai đoạn và đa tác tử tập trung. Mục
tiêu là so sánh công bằng, không điều chỉnh thí nghiệm để buộc đa tác tử phải
đạt kết quả cao nhất.

## Quy trình vận hành

1. Tải lên một BRD/SRS có thể trích xuất văn bản và ghi nhận `document_hash`.
2. Với mỗi cặp `(document_hash, evaluator_version)`, tác tử Judge chỉ trích xuất
   một danh mục coverage. `evaluator_version` bao gồm phiên bản prompt, schema,
   nhà cung cấp, mô hình và mức thinking.
3. Kiểm tra từng coverage unit cùng trích dẫn nguyên văn, số trang, mục và
   `chunk_id`. Nếu cần sửa nội dung thì phải tạo phiên bản evaluator và danh mục
   mới; không chỉnh sửa snapshot tại chỗ.
4. Phê duyệt danh mục trước khi đưa các run vào so sánh chính thức.
5. Chạy các điều kiện trên cùng tài liệu và ánh xạ test case của từng run vào
   đúng `catalog_id` đã phê duyệt. Không trích xuất lại mẫu số recall cho từng
   điều kiện.
6. Lưu bất biến danh mục, mapping, score hoặc trạng thái lỗi. Lỗi Judge không
   được chuyển thành F1 bằng 0 và không làm thay đổi kết quả validation của quá
   trình sinh.
7. Hai người đánh giá độc lập, không xem nhãn điều kiện, chấm bốn chiều:
   coverage, groundedness, executability và redundancy control trên thang 1–4.
8. Lưu phán quyết adjudication riêng sau khi hai lượt chấm độc lập hoàn tất;
   không ghi đè điểm ban đầu.
9. Báo cáo exact agreement, adjacent agreement và quadratic-weighted Cohen's
   kappa (QWK) theo từng chiều. Ngưỡng vận hành là QWK ≥ 0,70 cho cả bốn chiều.

Ngưỡng 0,70 là cổng chất lượng của dự án, không phải nhãn diễn giải phổ quát.
Khi một chiều có ít hơn hai cặp điểm, hệ thống báo thiếu dữ liệu. Khi mọi điểm
đều giống hệt nhau và không có biến thiên biên, kappa không xác định; trường hợp
này cũng không được xem là đạt cổng.

## Ý nghĩa của semantic coverage F1

Đây là **semantic coverage F1 dành riêng cho dự án**, không phải classifier
micro-F1.

- **Precision test case** là tỷ lệ test case được Judge ánh xạ tới ít nhất một
  coverage unit hợp lệ: `TP / (TP + FP)`.
- **Recall coverage unit** là tỷ lệ coverage unit được ít nhất một test case bao
  phủ: `TP_unit / tổng coverage unit`, tương đương
  `(tổng coverage unit - FN) / tổng coverage unit`.
- **F1** là trung bình điều hòa `2PR / (P + R)`; bằng 0 khi cả precision và
  recall đều bằng 0.

Một test case không xuất hiện trong batch mapping hoặc có danh sách coverage
unit rỗng được tính là false positive. Coverage unit không được bất kỳ test case
nào bao phủ được tính là false negative. Mapping trùng lặp cho cùng test case bị
từ chối thay vì dùng quy tắc “giá trị cuối thắng”. ID ngoài catalog hoặc ngoài
test suite không được dùng để tăng điểm.

## Tính bất biến và khả năng kiểm toán

- Khóa duy nhất `(document_hash, evaluator_version)` bảo đảm mọi điều kiện dùng
  cùng một snapshot coverage.
- `coverage_scores.catalog_id` liên kết score mới với danh mục; score cũ được
  giữ dưới nhãn `legacy-unversioned` để tương thích dữ liệu lịch sử.
- Mỗi evaluation lưu mapping JSONB đầy đủ, thời điểm đánh giá và trạng thái
  `completed` hoặc `failed`.
- Rating và adjudication là dữ liệu append-only. Khóa duy nhất ngăn một người
  chấm ghi đè cùng chiều và cùng vòng.
- Run lưu cấu hình mô hình, prompt, token ceiling, token đã dùng và các artifact
  nguồn; credential và PDF thô không được lưu.

## Diễn giải và giới hạn

F1 tự động đo mức bao phủ so với một catalog do LLM tạo và ánh xạ, vì vậy không
thay thế đánh giá chuyên gia. Catalog đóng băng làm tăng tính so sánh giữa các
điều kiện nhưng không chứng minh catalog hoàn hảo. Việc phê duyệt trích dẫn,
đánh giá mù bởi hai người và adjudication giảm rủi ro này.

Temperature bằng 0 không bảo đảm đầu ra của dịch vụ LLM luôn hoàn toàn xác
định. Báo cáo cuối phải nêu số lần lặp, token thực tế, lỗi Judge bị loại khỏi
phép so sánh và mọi thay đổi `evaluator_version`. Không diễn giải kết quả hiện
tại như bằng chứng mặc định rằng đa tác tử tốt hơn; đó là giả thuyết cần kiểm
định trên tài liệu giữ lại.

## Cơ sở nghiên cứu công khai

- Cohen's weighted kappa hỗ trợ mức phạt theo khoảng cách giữa các hạng mục thứ
  bậc; dự án dùng trọng số bình phương cho thang 1–4:
  <https://pubmed.ncbi.nlm.nih.gov/19673146/>.
- Nghiên cứu LLM-as-a-Judge cho thấy bias theo vị trí và các sai lệch đánh giá,
  tạo cơ sở cho việc đóng băng evaluator, làm mù nhãn điều kiện và hiệu chỉnh
  bằng con người:
  <https://aclanthology.org/2024.emnlp-main.474/> và
  <https://aclanthology.org/2025.ijcnlp-long.18/>.
- Multi-agent debate không mặc nhiên vượt self-consistency hoặc ensemble nếu
  protocol không được kiểm soát, vì vậy giả thuyết đa tác tử phải được kiểm
  chứng thay vì giả định:
  <https://proceedings.mlr.press/v235/smit24a.html>.
- Việc chọn kiểm định thống kê phải phù hợp thiết kế ghép cặp và phân phối dữ
  liệu, không chọn tùy ý một phép kiểm định:
  <https://aclanthology.org/P18-1128/>.
- Temperature 0 vẫn có thể cho kết quả khác nhau trên hệ thống LLM được phục
  vụ, nên cần lặp run và báo cáo biến thiên:
  <https://arxiv.org/abs/2410.03492>.

## Kiểm tra tái lập

```sh
env PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_coverage.py tests/test_evaluation.py tests/test_runner.py tests/test_app.py
env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test \
  .venv/bin/python -m pytest -q tests/test_storage.py
git diff --check
```

Chỉ báo cáo kết quả thí nghiệm chính thức khi danh mục đã được phê duyệt, mọi
điều kiện cùng `catalog_id`, lỗi evaluator đã bị loại khỏi so sánh và QWK đạt
ngưỡng ở từng chiều hoặc được báo rõ là chưa đạt.
