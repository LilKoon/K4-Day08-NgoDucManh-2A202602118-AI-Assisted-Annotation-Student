# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: **Ngô Đức Mạnh**  
Công cụ gán nhãn: **CVAT Docker local**, import pre-label và export Ultralytics YOLO Detection 1.0.  
Nguồn số liệu: `outputs/metrics_round0.json`, `outputs/metrics_round1.json`, `outputs/round1_diff.md`, `outputs/selection_round1.csv`, `reports/rounds_table.md` và hai ảnh `outputs/compare_round*.jpg`.

## 1. Dữ liệu và cách chia tập

Video là một cảnh camera cố định ở đường cao tốc ban đêm. Các frame cách nhau 0,4 giây nên cùng một xe có thể xuất hiện trong nhiều ảnh liên tiếp. Nếu chia ngẫu nhiên, ảnh train và test dễ chứa cùng xe, nền đường và điều kiện ánh sáng gần như giống nhau; số đo test khi đó có xu hướng **cao hơn khả năng tổng quát thực tế**. Dữ liệu được chia theo thời gian thành 268 ảnh pool, 20 ảnh test và 112 ảnh vùng đệm/bị loại. Ảnh pool gần ảnh test nhất vẫn cách 4,4 giây (`data/DATA.md`). Không dùng ảnh hoặc nhãn test để huấn luyện.

Tập test có 417 box tham chiếu; phép chấm bỏ qua 14 box cao dưới 16 pixel, còn 403 box được tính. Các box tham chiếu do mô hình tạo và chưa được người rà từng box, nên AP50 ở đây đo mức khớp với bộ tham chiếu đó.

## 2. Mô hình khởi đầu lạnh (cold start)

Mô hình `yolov8n.pt` chưa fine-tune trên video này đạt **AP50 = 0,7714** trên 20 ảnh test. Tại confidence 0,25, nó có TP=197, FP=16, FN=206, precision=0,9249 và recall=0,4888 (`outputs/metrics_round0.json`). Độ phủ theo cỡ xe là **0,1818 cho xe nhỏ** (66 box tham chiếu), **0,5473 cho xe vừa** (296 box) và **0,5610 cho xe lớn** (41 box). Như vậy xe nhỏ là nhóm yếu rõ nhất theo bộ tham chiếu.

Trong `outputs/compare_round0.jpg`, `frame_0050.jpg` có 11 TP, 2 FP, 7 FN; `frame_0350.jpg` có 9 TP, 2 FP, 14 FN. Ảnh đêm có xe xa chỉ hiện đèn nhỏ, xe tối hoặc bị che một phần; đây là những vị trí cần kiểm thủ công trước khi kết luận mọi FN đều là lỗi của mô hình. Chẳng hạn một cụm hai chấm đèn sát đường chân trời trong `frame_0350.jpg` cần người rà xem có đủ dấu hiệu là xe và box tham chiếu có đúng phạm vi hay không. Tôi không sửa nhãn test để làm đẹp số đo.

## 3. Chiến lược chọn mẫu

Notebook xếp hạng ảnh pool theo `score = 0,5·U + 0,3·A + 0,2·D`: `U` là trung bình độ bất định của năm box khó nhất, `A` là số box có confidence mơ hồ được chuẩn hóa, `D` biểu diễn khoảng cách thời gian tới ảnh đã gán nhãn. Trong 50 ứng viên đầu vòng 1, `D=1,0` cho mọi ảnh nên `U` và `A` phân biệt thứ hạng nhiều hơn. `MIN_GAP_S=2,0` yêu cầu hai ảnh trong lô cách nhau ít nhất 2 giây để hạn chế lấy liên tiếp gần như cùng một cảnh.

Ba ảnh lô 12 được phân tích trong `reports/SELECTION.md` là `frame_0182.jpg` (hạng 1, điểm 0,9591, 18 box mơ hồ), `frame_0369.jpg` (hạng 2, 0,9324, 16 box mơ hồ) và `frame_0326.jpg` (hạng 4, 0,9155, 15 box mơ hồ). `frame_0372.jpg` cũng đạt 0,9101 nhưng không được chọn vì chỉ cách `frame_0369.jpg` 1,2 giây. Nếu chỉ có công rà năm ảnh, tôi chọn các thời điểm 39,6; 72,8; 90,8; 130,4 và 147,6 giây để phân tán công, thay vì lấy thêm `frame_0331.jpg` cách `frame_0326.jpg` đúng 2 giây hoặc `frame_0380.jpg` cách `frame_0369.jpg` 4,4 giây. Box dự đoán nhiều hơn cũng làm công rà tăng. Điểm bất định **không chứng minh** ảnh đó sẽ cải thiện mô hình; kết quả vòng 1 dưới đây là phản ví dụ cho việc suy từ điểm chọn mẫu sang AP50.

## 4. Các vòng học chủ động (active learning)

| Vòng | Ảnh train | Box train | AP50 test | Δ so với cold start | Precision @0,25 | Recall @0,25 | Recall nhỏ / vừa / lớn |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0,7714 | — | 0,9249 | 0,4888 | 0,1818 / 0,5473 / 0,5610 |
| 1 | 12 | 333 | 0,3079 | **−0,4635** | 1,0000 | 0,0223 | 0,0000 / 0,0068 / 0,1707 |

Vòng 1 là vòng fine-tune đầu tiên nên chênh lệch so với vòng trước cũng là **−0,4635 AP50**. Theo phép so box ở `outputs/round1_diff.md`, AI gợi ý ban đầu có 169 box trên 12 ảnh. Bản nhãn export từ CVAT có 333 box: **114 accepted, 38 edited, 17 deleted và 181 added**. Đây là thống kê đối chiếu theo IoU, không thay cho việc nhìn từng box để đánh giá đúng/sai. `outputs/metrics_round1.json` xác nhận mô hình vòng 1 được huấn luyện trên đúng 12 ảnh và 333 box, 50 epoch; số đo trong bảng là trên **20 ảnh test**, không phải số mAP in cho ảnh train.

Quan sát độc lập đã khóa ở `reports/BLIND_SCAN.md` ghi 26 xe trong `frame_0099.jpg`, đặc biệt lưu ý xe xa và xe sát tường bên trái bị che. Bản nhãn sau rà của frame này cũng có 26 box, nhưng số lượng bằng nhau không tự chứng minh từng box khớp. `reports/REVIEW_LOG.csv` ghi ba trường hợp thêm box cụ thể ở frame này; phép diff toàn frame ghi 8 accepted, 5 edited và 13 added. Các con số đó mô tả **pre-label được sửa**, không phải dự đoán của mô hình sau fine-tune.

Sau fine-tune, kết quả test giảm mạnh: TP từ 197 xuống **9**, FN từ 206 lên **394** tại confidence 0,25; FP vòng 1 bằng 0. Precision 1,0000 chỉ tính trên rất ít dự đoán vượt ngưỡng nên không có nghĩa mô hình tốt hơn. Recall xe nhỏ giảm về 0, xe vừa còn 0,0068 và xe lớn còn 0,1707. Trong `outputs/compare_round1.jpg`, `frame_0050.jpg` đổi từ 11 TP, 2 FP, 7 FN ở cold start sang 0 TP, 0 FP, 18 FN ở vòng 1; `frame_0150.jpg` từ 10 TP, 2 FP, 10 FN sang 0 TP, 0 FP, 20 FN. Đây là bằng chứng trực quan cho thấy mô hình gần như không xuất box ở ngưỡng đang dùng. AP50 vẫn là 0,3079 vì phép tính AP xét xếp hạng dự đoán qua nhiều mức confidence, còn TP/FP/FN ở đây cố định tại 0,25.

Nguyên nhân giảm chưa được xác định chỉ từ hai file metrics. Những khả năng cần kiểm là chất lượng và tính nhất quán của 333 box đã export, phân bố confidence sau train, cấu hình fine-tune và việc 12 frame cùng một cảnh có nhiều xe gần trùng. Ca khó là xe bị che ở góc cua: theo `GUIDELINE_LABEL.md`, chỉ vẽ phần **nhìn thấy** khi còn nhận ra đó là xe, không suy đoán phần bị che và không ôm vệt đèn trên đường. Trước khi huấn luyện thêm, cần đối chiếu ảnh với nhãn export ở các ca này và xem dự đoán ở vài ngưỡng confidence để tìm nguyên nhân recall sụt.

## 5. Kết luận và giới hạn

Sau một vòng sửa nhãn và fine-tune, mô hình **kém hơn cold start theo bộ test hiện có**: AP50 giảm 0,4635 và recall @0,25 giảm từ 0,4888 xuống 0,0223. Tôi dừng sau vòng bắt buộc để kiểm tra sự sụt giảm thay vì lập tức lấy thêm ảnh. Ưu tiên rà lại nhãn export của 12 ảnh, nhất là 181 box được thêm theo phép diff, box xe bị che và xe xa; sau đó kiểm tra phân bố confidence và cấu hình train. Việc đóng gói đã xác nhận đủ 12 file nhãn YOLO và 333 box, nhưng kiểm định dạng không chứng minh mọi box đều chính xác.

Nếu làm vòng sau, hai nhóm đáng xem là **xe nhỏ/xa trong luồng xe đông** (ví dụ `frame_0350.jpg`, còn 22 FN ở vòng 1 trong ảnh so sánh) và **xe bị che tại góc cua hoặc cạnh vật chắn**. Mỗi nhóm cần ảnh đủ khác nhau theo thời gian, tránh rà nhiều frame của cùng xe trong cảnh camera cố định. Công rà nhãn cũng đáng kể: ảnh được chọn có thể có 28–43 box dự đoán và số box cuối cùng còn nhiều hơn. Chỉ thêm dữ liệu khi đã hiểu nguyên nhân lỗi hiện tại; không có bảo đảm vòng 2 sẽ tăng AP50.

Kết luận chỉ dựa trên 20 ảnh test, luật bỏ qua 14 box cao dưới 16 pixel và nhãn tham chiếu do mô hình khác tạo, chưa được người rà từng box. Với tập nhỏ, chênh lệch rất bé có thể là nhiễu; ở đây mức giảm lớn nên cần điều tra, nhưng vẫn chưa đủ để quy nguyên nhân cho một thao tác gán nhãn cụ thể hoặc suy ra chất lượng trên mọi video đường đêm.
