# YOLO OBJECT DETECTION SYSTEM
## 📍 Đề tài
**Xây dựng hệ thống ứng dụng mô hình Deep Learning YOLO để phát hiện đối tượng trong ảnh**

**BUILDING A SYSTEM USING YOLO DEEP LEARNING MODEL TO DETECTING OBJECTS IN IMAGES**

## 👨‍🏫 Giảng viên hướng dẫn
- ThS. Võ Quang Hoàng Khang

## 📚 Môn học
- Thị giác máy tính và ứng dụng - Computer vision

## 👨‍🎓 Nhóm sinh viên thực hiện
| Họ và Tên      | MSSV      |
|----------------|-----------|
| Nguyễn Phú Sang| 21023391  |
| Lưu Chí Tài    | 21080681  |

## 🏫 Trường
- **Trường Đại học Công nghiệp TP. Hồ Chí Minh**
- **Khoa Công nghệ Thông tin**

## 📦 Mô tả dự án

Đề tài tập trung vào việc xây dựng hệ thống phát hiện đối tượng trong ảnh sử dụng mô hình học sâu YOLOv8, nhằm giải quyết các bài toán thực tế trong thị giác máy tính như giám sát an ninh, giao thông, xe tự hành,...

Hệ thống được xây dựng bằng Python, sử dụng các thư viện:
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Tkinter](https://docs.python.org/3/library/tkinter.html) (giao diện)

## 🎯 Mục tiêu
- Nghiên cứu lý thuyết về Object Detection, đặc biệt là YOLOv8
- Huấn luyện mô hình phát hiện đối tượng trên tập dữ liệu Pascal VOC 2012
- Xây dựng ứng dụng có giao diện trực quan sử dụng mô hình đã huấn luyện
- Đánh giá kết quả dựa trên các chỉ số: `mAP@0.5`, `Precision`, `Recall`, `F1-score`, `Confusion Matrix`

## 🧪 Môi trường thực nghiệm
- Nền tảng: Google Colab
- GPU: NVIDIA Tesla T4
- Python: 3.10

## 📂 Dataset
- **Tên:** Pascal VOC 2012
- **Số lượng ảnh:** 17,125 ảnh
- **Số lớp:** 20 lớp (person, car, cat, dog,...)

## 🧠 Mô hình sử dụng
- YOLOv8 (phiên bản YOLO hiện đại, anchor-free)
- Huấn luyện với các tham số:
  - Epochs: 5
  - Batch size: 32
  - Image size: 320x320
  - Pretrained weights: `yolov8n.pt`

## 📈 Kết quả huấn luyện
| Chỉ số            | Giá trị     |
|-------------------|-------------|
| Precision         | 0.7122      |
| Recall            | 0.5940      |
| mAP@0.5           | 0.6531      |
| mAP@0.5:0.95      | 0.4890      |
| F1-score          | 0.6478      |

## 🖥️ Giao diện hệ thống
- Được xây dựng bằng `Tkinter`, có 2 phần chính:
  - Màn hình đăng nhập
  ![image](https://github.com/user-attachments/assets/aab6445c-aa52-4063-a83f-f537022a546f)
  - Màn hình chính: hiển thị ảnh/webcam, bảng kết quả (label, confidence, bbox,...), các chức năng như mở webcam, lưu file CSV, dark mode,...
  ![image](https://github.com/user-attachments/assets/fc9aae87-523c-461e-9062-22a9b4fadf62)

## 🚀 Hướng phát triển
- Sử dụng mô hình nâng cao như `YOLOv8x`, `YOLOv8-seg` để kết hợp segmentation
- Tăng cường dữ liệu với kỹ thuật `Mosaic`, `CutMix`
- Khắc phục mất cân bằng lớp bằng `Focal Loss`
- Áp dụng vào các bài toán thực tế như: theo dõi giao thông, giám sát CCTV, nông nghiệp thông minh

## 📚 Tài liệu tham khảo
- Redmon, J., et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*
- Bochkovskiy, A., et al. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection*
- Allied Market Research Report on Computer Vision Market, 2021–2030

---

🎉 *Cảm ơn bạn đã đọc README này. Hệ thống có thể triển khai thực tế trong các dự án giám sát và xử lý ảnh/video thời gian thực!*
