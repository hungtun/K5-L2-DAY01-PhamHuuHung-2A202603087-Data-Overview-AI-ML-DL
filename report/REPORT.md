# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU (`device = cpu`)

**Python / PyTorch / Ultralytics:** Python `3.12.0` / PyTorch `2.15.0` / Ultralytics `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (giữ nguyên mặc định của bài lab, có đối chiếu thêm các mức `0.20` và `0.60` ở phần phát hiện vật thể)

**Thay đổi so với notebook nguồn:** Không (giữ nguyên phiên bản thư viện `ultralytics==8.4.145`, 3 checkpoint cùng mã băm SHA-256, 3 ảnh mẫu COCO `traffic`, `kitchen`, `dining` và ngưỡng `0.35`)

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Chép record hạng 1:**
   - `sample_id` = `traffic` (`coco_image_id` = `210273`, kích thước `640 × 428`)
   - `class_id` = `468` ; `class_name` = `cab` ; `rank` = `1` ; `score` = `0.510915` ; `taxonomy_name` = `ImageNet-1K` (`model_file` = `yolo11n-cls.pt`)
2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**
   - Record này mô tả nhãn cho **toàn bộ bức ảnh (image-level classification)**, không phải cho một vật thể riêng lẻ.
   - **Lý do:** Trong JSON của `classification_predictions.json`, trường `task` là `"image_classification"` và bản ghi chỉ chứa kích thước chung của ảnh (`image_width = 640`, `image_height = 428`) cùng thứ hạng dự đoán (`rank`, `class_id`, `class_name`, `score`), hoàn toàn **không có bất kỳ trường tọa độ không gian nào** (`bbox_xyxy` hay `polygon_xy`) để định vị một đối tượng cụ thể trong khung hình.
3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**
   - Danh sách tên lớp **đi kèm cố định với taxonomy của tập dữ liệu huấn luyện và được đóng gói sẵn trong file checkpoint** (`yolo11n-cls.pt` sử dụng taxonomy `ImageNet-1K` gồm 1.000 lớp cố định). Mô hình chỉ tính xác suất trên 1.000 lớp có sẵn này chứ **không tự nghĩ ra** tên lớp mới.
4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**
   - `taxonomy_name` (`ImageNet-1K`) xác định hệ quy chiếu bộ nhãn đang dùng, vì hai bộ dữ liệu khác nhau có tập lớp và quy ước hoàn toàn khác nhau (ví dụ `ImageNet-1K` khác `COCO-80`).
   - `class_id` (`468`) là mã định danh số nguyên cố định giúp chương trình máy tính ánh xạ chính xác tuyệt đối, tránh lỗi do đồng âm, khác biệt chữ hoa/thường, dấu gạch dưới hay khi đổi tên hiển thị (ví dụ `class_id = 0` trong `COCO-80` là `person`, nhưng trong `ImageNet-1K` lại là `tench`).
   - `class_name` (`cab`) giúp con người (annotator, reviewer, kỹ sư) đọc hiểu trực tiếp ngữ nghĩa của lớp khi kiểm tra báo cáo hoặc hình minh họa mà không phải tra cứu bảng mã.
5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**
   - Trên ảnh `traffic` có rất nhiều đối tượng cùng xuất hiện (xe buýt xanh/trắng, nhiều ô tô con, xe tải nhỏ, người đi bộ trên vỉa hè). Nếu bài toán yêu cầu phân loại đơn nhãn (single-label), guideline bắt buộc phải quy định rõ **tiêu chí chọn chủ đề chính (primary subject rule)**:
     - Ưu tiên đối tượng chiếm diện tích lớn nhất hoặc nằm ở vị trí trung tâm/tiền cảnh;
     - Hoặc quy định thứ tự ưu tiên rõ ràng giữa các nhóm lớp khi cùng xuất hiện trong một cảnh giao thông;
     - Hoặc định nghĩa rõ khi nào ảnh đa chủ thể phải gán nhãn cảnh tổng quát / đưa vào diện không hợp lệ hay escalation thay vì chọn đại một loại phương tiện như `cab`.
6. **Vì sao không được lấy `score` của model làm ground truth?**
   - `score` (`0.510915` cho lớp `cab` trên ảnh `traffic`, hay `0.420191` cho lớp `gong` trên ảnh `kitchen`) chỉ là độ tự tin toán học (softmax probability) của mô hình khi khớp đặc trưng ảnh với taxonomy `ImageNet-1K`. Mô hình có thể có `score` cao nhất ở `rank = 1` nhưng vẫn dự đoán sai bản chất bức ảnh (ảnh `traffic` là cảnh đường phố nhiều làn xe chứ không phải một chiếc taxi `cab` đơn lẻ; ảnh `kitchen` là nhà bếp nhưng bị đoán hạng 1 là cái chiêng `gong` do các chảo đồng tròn treo trên tường). Ground truth bắt buộc phải do con người xác lập và kiểm chứng theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record (`sample_id = kitchen`, `coco_image_id = 397133`, `score_threshold = 0.35`):**
   - `class_id` = `0` ; `class_name` = `person` ; `score` = `0.912624` ; `bbox_xyxy` = `[385.33, 69.24, 498.92, 348.92]` ; `bbox_width` = `113.58` ; `bbox_height` = `279.68` (`coordinate_unit` = `pixel`, `bbox_format` = `xyxy`, `taxonomy_name` = `COCO-80`).
2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**
   - Trên ảnh `kitchen` có kích thước `640 × 427` pixel với gốc tọa độ `(0, 0)` ở góc trên-trái của bức ảnh:
     - **Góc trên-trái** của hộp nằm tại tọa độ `(x_min, y_min) = (385.33, 69.24)` — tức cách mép trái ảnh `385.33` pixel và cách mép trên ảnh `69.24` pixel (ngang đỉnh đầu người đầu bếp).
     - **Góc dưới-phải** của hộp nằm tại tọa độ `(x_max, y_max) = (498.92, 348.92)` — tức cách mép trái ảnh `498.92` pixel và cách mép trên ảnh `348.92` pixel (ngang bàn chân người đầu bếp).
     - **Kích thước hộp:** rộng `bbox_width = 113.58` pixel (`498.92 - 385.33`) và cao `bbox_height = 279.68` pixel (`348.92 - 69.24`).
3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**
   - Với ảnh `sample_id = kitchen`:
     - Tại ngưỡng **`0.20`**: có **17** prediction (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']`).
     - *(Tại ngưỡng mặc định **`0.35`** tạo file JSON: có **11** prediction: `['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl']`).*
     - Tại ngưỡng **`0.60`**: có **6** prediction (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person']`).
4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**
   - Khi hạ ngưỡng từ `0.60` (6 hộp) hoặc `0.35` (11 hộp) xuống `0.20` (17 hộp), số lượng prediction tăng lên đáng kể nhưng kéo theo nhiều **dương tính giả (false positives - bắt nhầm vật thể hoặc gán sai lớp)** và **hộp chồng lấn/lệch biên**.
   - Người **reviewer** (cùng annotator nếu dùng pre-annotation) phải làm thêm các việc:
     - Duyệt từng hộp mới xuất hiện để **xóa các hộp dự đoán sai** (ví dụ bó thảo mộc/lá khô treo ở góc trên-trái bị đoán nhầm là `potted plant`, hoặc hũ/vật dụng bị đoán nhầm là `bottle`);
     - Kiểm tra và **tinh chỉnh lại biên hộp (resize/tighten)** cho các vật thể đúng nhưng hộp bị lệch hoặc thừa/thiếu diện tích;
     - Đồng thời vẫn phải kiểm tra các **âm tính giả (false negatives - vật thể thật bị bỏ sót)** vì hạ ngưỡng vẫn không bắt hết các vật thể khó trong ảnh.
5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).**
   - **Quy tắc đóng khung vật thể (Tight Bounding Box Rule):** *"Mỗi vật thể thuộc lớp mục tiêu phải được bao bằng một hộp chữ nhật riêng biệt (`bbox_xyxy`) vừa khít với toàn bộ phần diện tích nhìn thấy được (visible pixels) của vật thể đó, độ lệch biên không quá 2 pixel ở cả 4 cạnh; không bao gồm bóng đổ hay nền thừa, và chỉ gán nhãn khi vật thể lộ diện tối thiểu 20% diện tích với kích thước tối thiểu từ 10 × 10 pixel trở lên."*
6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**
   - **Quan sát trên ảnh `kitchen` (`visuals/detection_predictions.png` & JSON):**
     - Ở sát mép trái ảnh có một **bàn tay/cánh tay người bị cắt mép** tại `bbox_xyxy = [0.12, 263.18, 61.55, 310.78]` được model dự đoán là `person` (`score = 0.61093`), và ngay bên dưới là một **khuôn/tô bánh bị cắt mép trái** tại `bbox_xyxy = [0.32, 305.02, 87.69, 371.32]` (`bowl`, `score = 0.381214`). Ngoài ra, trên bàn có hai chiếc ca/cốc kim loại (`cup` tại `[119.95, 272.2, 141.78, 304.16]` và `[143.52, 268.13, 172.93, 301.81]`) đứng sát và che khuất một phần nhau.
   - **Hướng xử lý và lý do:**
     - Đối với các vật thể **bị che khuất một phần hoặc cắt mép thông thường** (như chiếc tô `bowl` ở mép trái hay hai chiếc `cup` đứng cạnh nhau): nên **đưa thành quy tắc chuẩn trong guideline** (quy định ngưỡng phần trăm diện tích nhìn thấy tối thiểu, ví dụ ≥ 20%, và yêu cầu kéo cạnh box dừng đúng tại biên ảnh) vì đây là tình huống lặp lại thường xuyên trong dữ liệu thực tế, giúp đội ngũ annotator làm việc thống nhất.
     - Tuy nhiên, với trường hợp **chỉ lộ một bàn tay/cổ tay rất nhỏ ở sát mép ảnh (`[0.12, 263.18, 61.55, 310.78]`, chiếm chưa tới 10% cơ thể)** mà guideline chưa nêu rõ có tính là một instance `person` hợp lệ hay bỏ qua, annotator cần **đưa lên hỏi Lab Coach (escalation)** để chốt quy định trước khi gán nhãn hàng loạt, tránh tự suy diễn gây nhiễu tập ground truth.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Chép một record (`sample_id = kitchen`, `coco_image_id = 397133`, `score_threshold = 0.35`):**
   - `instance_id` = `kitchen-001` ; `class_id` = `0` ; `class_name` = `person` ; `score` = `0.899318` ; số điểm polygon = `polygon_point_count` = `348` ; 5 điểm đầu của `polygon_xy` = `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]` (đi kèm `bbox_xyxy` = `[385.45, 66.44, 498.02, 348.58]`).
2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**
   - Hộp giới hạn (`bbox_xyxy`) chỉ là hình chữ nhật 4 cạnh nên chứa nhiều vùng pixel nền không thuộc vật thể (ví dụ khoảng trống giữa hai chân, vùng tủ bếp quanh đầu/vai của `person` `kitchen-001`, hoặc góc chéo rất lớn của chiếc bàn `dining table` `kitchen-005`).
   - Ngược lại, **polygon (`polygon_xy`)** dùng tập hợp nhiều điểm tọa độ `[x, y]` (`348` điểm với `kitchen-001`) nối liên tiếp bám sát **đường viền thực tế (contour)** của đối tượng. Nhờ đó, polygon cung cấp chính xác hình dáng hình học, đường cong, hướng nghiêng và diện tích pixel thực sự của vật thể, tách biệt hoàn toàn pixel của đối tượng với nền và các vật thể xung quanh.
3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**
   - **Công dụng của `instance_id`:** Dùng để phân biệt và định danh duy nhất từng **cá thể (instance)** vật thể riêng lẻ trong phạm vi kết quả của một bức ảnh (ví dụ trong ảnh `kitchen` có 4 chiếc tô cùng thuộc lớp `bowl` nhưng được gán 4 mã cá thể riêng: `kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010`).
   - **Không phải `class_id`:** `class_id` định danh **loại lớp** trong taxonomy (mọi chiếc tô đều có chung `class_id = 45`), còn `instance_id` định danh **từng vật thể cụ thể** (hai vật cùng lớp vẫn phải có hai `instance_id` và hai đa giác riêng).
   - **Không phải `tracking_id`:** `tracking_id` được dùng trong bài toán video/chuỗi khung hình (Multi-Object Tracking) để duy trì cùng một mã định danh cho một đối tượng xuyên suốt các frame theo thời gian. Còn `instance_id` có dạng `<sample_id>-NNN` (`kitchen-001`) chỉ là mã đánh số thứ tự cục bộ trong một ảnh tĩnh của bài lab, không có mối liên hệ định danh đối tượng giữa ảnh này với ảnh khác.
4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo.**
   - **Quy tắc vẽ đa giác phân đoạn (Instance Polygon Rule):** *"Đặt các đỉnh đa giác (`polygon_xy`) bám sát ranh giới vật lý nhìn thấy được của từng cá thể (sai số không quá 1–2 pixel so với biên thật), đặt nhiều điểm hơn ở các đường cong/góc gập và ít điểm hơn ở cạnh thẳng; mỗi cá thể cùng lớp phải được vẽ thành một instance độc lập, không gộp nhiều vật thể rời nhau vào chung một polygon và không để các mask của những vật thể liền kề chồng lấn lên nhau."*
5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**
   - **Quan sát trên ảnh `kitchen` (`visuals/segmentation_prediction.png` & JSON):**
     - Trên mặt bàn gỗ (`dining table`, `instance_id = kitchen-005`, `polygon_point_count = 598`), các khuôn/tô bánh (`bowl` `kitchen-002`, `kitchen-003`, `kitchen-006`) nằm sát chạm nhau và đè lên mặt bàn; mask dự đoán của `dining table` (`kitchen-005`) bị lem/đè trùm lên nhiều đồ vật trên bàn (ca kim loại, cây lăn bột, đĩa bánh) và ở cạnh trên bên trái thì biên giữa mặt bàn với bếp lò màu đen (`oven` `kitchen-007`) rơi vào vùng tối có độ tương phản rất thấp khiến biên mask bị lẹm và nhập nhằng. Ngoài ra, ở góc trên-trái, bó lá khô treo trên trần (`kitchen-004`, `score = 0.632236`) bị model phân đoạn thành `potted plant` dù không có chậu cây.
   - **Hướng xử lý và lý do:**
     - Với trường hợp **hai vật chạm nhau hoặc vật đặt trên mặt bàn (`bowl` đặt trên `dining table`)**: áp dụng **guideline** nếu tài liệu đã quy định rõ quy tắc ưu tiên lớp tiền cảnh/hậu cảnh (z-order) và cách vẽ biên tiếp giáp giữa hai instance chạm nhau.
     - Với trường hợp **ranh giới giữa bàn (`kitchen-005`) và bếp lò đen (`kitchen-007`) chìm trong vùng tối không rõ biên vật lý, hoặc trường hợp bó thảo mộc treo (`kitchen-004`) có được gán là `potted plant` hay không**: cần **escalation (hỏi Lab Coach)** vì đây là vùng biên mơ hồ / ngoại lệ ngữ nghĩa mà annotator không được tự ý đoán biên hoặc tự bịa nhãn.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một (hoặc nhiều) nhãn lớp cấp toàn ảnh (`class_id`, `class_name`) chọn từ taxonomy cố định theo guideline, không có tọa độ vùng. | Ảnh `traffic` có nhiều xe buýt, ô tô, người đi bộ nhưng model (`yolo11n-cls.pt`, `ImageNet-1K`) đoán `rank=1` là `cab` (`score=0.510915`); ảnh `kitchen` bị đoán `rank=1` là `gong` (`score=0.420191`) do các chảo đồng tròn treo tường. | Đọc kỹ quy tắc chọn chủ đề chính trong guideline để gán đúng lớp đại diện cho toàn ảnh; nếu ảnh đa chủ thể hoặc ngoài phạm vi taxonomy thì đánh dấu/escalate thay vì chép theo `rank=1` của model. | Kiểm tra nhãn cấp ảnh có khớp đúng chủ đề chính theo guideline và đúng `taxonomy_name` hay không; phát hiện lỗi gán nhãn bị ảnh hưởng bởi một chi tiết cục bộ gây nhiễu. |
| Phát hiện vật thể | Tập hợp các hộp giới hạn `bbox_xyxy = [x_min, y_min, x_max, y_max]` (pixel) kèm `class_id`/`class_name` cho **mọi** vật thể hợp lệ trong ảnh. | Trên ảnh `kitchen` (`detection_predictions.png`), model bắt một bàn tay bị cắt mép trái `[0.12, 263.18, 61.55, 310.78]` là `person` (`score=0.61093`), khi hạ ngưỡng xuống `0.20` thì bắt nhầm bó lá treo là `potted plant`, đồng thời bỏ sót nhiều xoong/chảo treo và muôi/thìa ở ngưỡng `0.35`. | Vẽ hộp vừa khít quanh từng vật thể thật theo guideline, bổ sung đầy đủ các vật thể bị model bỏ sót (false negatives) và loại bỏ các hộp dự đoán nhầm (false positives); hỏi Lab Coach với ca cắt mép cực đoan. | Kiểm tra đủ số lượng vật thể (không sót object), đúng `class_name`, độ khít của 4 cạnh hộp (`bbox_xyxy`), không có hộp trùng lặp và cách xử lý các vật bị che khuất/cắt mép đúng quy ước. |
| Instance segmentation | Tập hợp các cá thể độc lập (`instance_id`), mỗi cá thể có nhãn lớp và đa giác `polygon_xy` (danh sách điểm `[x, y]` bám sát biên từng vật thể). | Trên ảnh `kitchen` (`segmentation_prediction.png`), mask của `dining table` (`kitchen-005`, 598 điểm) tràn đè lên các đồ vật đặt trên bàn và nhập nhằng biên với `oven` (`kitchen-007`) trong vùng tối; bó lá khô góc trên-trái (`kitchen-004`) bị gán nhầm là `potted plant` (`score=0.632236`); hai chiếc `spoon` (`kitchen-008`, `kitchen-011`) có cán mảnh trong bóng tối nên biên mask bị lệch. | Vẽ đa giác `polygon_xy` bám sát biên thật của từng cá thể riêng biệt, tách rời các vật cùng lớp đứng sát nhau (như các `bowl` `kitchen-002`, `kitchen-003`, `kitchen-006`), xóa mask sai lớp và escalate vùng biên tối không nhìn rõ. | Kiểm tra độ chính xác của đường biên polygon (không lẹm vào vật thể, không tràn ra nền), đảm bảo mỗi vật thể là một instance riêng biệt không gộp chung hoặc chồng lấn mask, và xác nhận các ca khó đã được xử lý đúng quyết định từ Lab Coach. |

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)**
   - Chỉ sử dụng đúng 3 ảnh COCO công khai có kiểm tra mã băm SHA-256 kèm trích dẫn nguồn (`IMAGE_ATTRIBUTION.md`), tuyệt đối không tải ảnh cá nhân, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab/GitHub và không ghi họ tên, MSSV, email hay số điện thoại vào bên trong `REPORT.md` hoặc các tệp đầu ra.
2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**
   - Mình sẽ dừng ngay thao tác xử lý/tải lên, không chia sẻ hay commit dữ liệu đó lên kho mã nguồn công khai và báo ngay cho **Giảng viên / Lab Coach (hoặc Mentor phụ trách lớp)** để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
