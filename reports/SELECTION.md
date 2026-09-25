# Vì sao chọn lô ảnh này?

Nguồn: 50 ứng viên đứng đầu `outputs/selection_round1.csv` và ảnh tổng hợp `outputs/selection_round1.jpg`. Lô bắt buộc do mô hình chọn 12 ảnh; bảng dưới là **quyết định giả định nếu chỉ đủ công rà 5 ảnh**, không thay đổi lô đã gán nhãn. `n_boxes` là số box mô hình dự đoán, chỉ là chỉ báo sơ bộ về công rà nhãn.

## Năm ảnh ưu tiên khi ngân sách là năm ảnh

| Ưu tiên | Frame | Hạng CSV | Giây | Điểm | Box dự đoán | Lý do |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `frame_0182.jpg` | 1 | 72,8 | 0,9591 | 28 | Điểm cao nhất, `A=1,0` với 18 box mơ hồ; ảnh tổng hợp cho thấy nhiều xe nhỏ và đèn lẫn nhau, trong khi số box cần rà thấp hơn nhiều ứng viên khác. |
| 2 | `frame_0369.jpg` | 2 | 147,6 | 0,9324 | 43 | `U=0,9315`, 16 box mơ hồ; cảnh đông xe về cuối video. Chi phí rà dự kiến cao nhưng có thêm nhóm xe và thời điểm khác với ảnh đầu. |
| 3 | `frame_0326.jpg` | 4 | 130,4 | 0,9155 | 39 | `U=0,9310`, 15 box mơ hồ; có xe ở nhiều khoảng cách và ánh đèn mạnh. Chọn một ảnh ở đoạn này thay vì rà thêm ảnh rất gần thời gian. |
| 4 | `frame_0099.jpg` | 8 | 39,6 | 0,9063 | 29 | `U=0,9460`, 14 box mơ hồ và công rà dự kiến thấp hơn các ảnh 39–47 box; cung cấp đoạn thời gian sớm hơn và có xe bị che/xe xa cần kiểm. |
| 5 | `frame_0227.jpg` | 11 | 90,8 | 0,8915 | 37 | Thêm khoảng thời gian ở giữa 72,8 và 130,4 giây, tránh dồn công vào hai cụm xe gần nhau; `U=0,9164`. |

Ảnh `frame_0380.jpg` đứng hạng 3, điểm 0,9170, nhưng cách `frame_0369.jpg` chỉ 4,4 giây trong cùng một cảnh quay camera cố định. `frame_0331.jpg` đứng hạng 5, điểm 0,9154, cách `frame_0326.jpg` đúng 2,0 giây. Chúng hợp lệ theo ngưỡng tự động 2 giây nhưng có nguy cơ lặp xe và bố cục; với ngân sách chỉ năm ảnh, tôi ưu tiên độ phủ thời gian. Đây là đánh đổi dựa trên nguy cơ gần trùng, không phải khẳng định hai cặp ảnh giống hệt nhau.

## Ba frame trong lô 12 ảnh được chọn

- `frame_0182.jpg` (hạng 1, điểm 0,9591): điểm cao chủ yếu từ 18 box mơ hồ và `A=1,0`; ảnh tổng hợp có nhiều xe với đèn pha sáng trên đường tối, cần rà các box gần nhau.
- `frame_0369.jpg` (hạng 2, điểm 0,9324): 43 box dự đoán và 16 box mơ hồ; ảnh cho thấy mật độ xe cao, nên nhiều tín hiệu để sửa nhãn nhưng cũng tốn công hơn.
- `frame_0326.jpg` (hạng 4, điểm 0,9155): `U=0,9310`, 15 box mơ hồ; ảnh có các xe ở cả gần và xa. Ảnh `frame_0331.jpg` cùng đoạn thời gian cũng được mô hình chọn dù chỉ cách 2 giây, vì vậy cần lưu ý độ trùng khi dùng ngân sách nhỏ.

## Một ứng viên điểm cao nhưng không được chọn

`frame_0372.jpg` đứng hạng 6 với điểm 0,9101, nhưng không thuộc lô 12 ảnh. Nó ở giây 148,8, chỉ cách `frame_0369.jpg` (147,6 giây) **1,2 giây**, thấp hơn `MIN_GAP_S=2,0`. Đây là ví dụ cụ thể về việc điểm cao vẫn phải nhường cho ràng buộc khoảng cách thời gian. `frame_0368.jpg` hạng 9 cũng chỉ cách `frame_0369.jpg` 0,4 giây.

## Điều phép chọn chưa chứng minh

Điểm dùng công thức `score = 0,5·U + 0,3·A + 0,2·D`: `U` là độ bất định của các box khó, `A` là lượng box có confidence mơ hồ, `D` là khoảng cách thời gian với ảnh đã gán nhãn. Trong 50 dòng đầu của vòng 1, `D=1,0` ở mọi dòng, nên thứ tự ở nhóm này chủ yếu do `U` và `A`. `MIN_GAP_S=2,0` giảm ảnh sát nhau nhưng không bảo đảm đa dạng thị giác. Điểm bất định không đo trực tiếp số lỗi nhãn, giá trị học được sau fine-tune hay mức cải thiện AP50; những điều đó phải kiểm bằng nhãn đã rà và đánh giá trên test.
