import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import os
import pandas as pd
from ultralytics import YOLO
import threading

import tkinter as tk
from tkinter import messagebox

# Danh sách user/password
users = {
    "sang": "123",
    "tai": "123"
}


def main_app():
    print("Đăng nhập thành công. Đây là giao diện chính!")

def login_window():
    login = tk.Tk()
    login.title("Đăng nhập hệ thống")
    login.state('zoomed')
    login.configure(bg="#f0f4f7")
    login.resizable(False, False)

    # Tiêu đề chính
    title_label = tk.Label(
        login,
        text="HỆ THỐNG NHẬN DIỆN ĐỐI TƯỢNG TRONG ẢNH/WEBCAM",
        font=("Arial", 24, "bold"),
        bg="#f0f4f7",
        fg="#0f4c75"
    )
    title_label.pack(pady=40)

    # Frame chứa khối đăng nhập
    frame = tk.Frame(login, bg="white", bd=2, relief="ridge")
    frame.place(relx=0.5, rely=0.5, anchor="center", width=500, height=350)

    # Label & Entry: Tên đăng nhập
    username_label = tk.Label(frame, text="👤 Tên đăng nhập:", font=("Arial", 14), bg="white")
    username_label.pack(pady=(25, 5))
    username_entry = tk.Entry(frame, font=("Arial", 14))
    username_entry.pack(pady=5, ipadx=50, ipady=5)

    # Label & Entry: Mật khẩu
    password_label = tk.Label(frame, text="🔒 Mật khẩu:", font=("Arial", 14), bg="white")
    password_label.pack(pady=(15, 5))
    password_entry = tk.Entry(frame, font=("Arial", 14), show="*")
    password_entry.pack(pady=5, ipadx=50, ipady=5)

    # Hàm xử lý đăng nhập
    def check_login():
        username = username_entry.get()
        password = password_entry.get()
        if username in users and users[username] == password:
            messagebox.showinfo("Thành công", "Đăng nhập thành công!")
            login.destroy()
            main_app()
        else:
            messagebox.showerror("Lỗi", "Tên đăng nhập hoặc mật khẩu không đúng!")

    # Nút đăng nhập
    login_button = tk.Button(
        frame,
        text="🔓 Đăng nhập",
        font=("Arial", 14, "bold"),
        command=check_login,
        bg="#3282b8",
        fg="white",
        activebackground="#0f4c75",
        width=15
    )
    login_button.pack(pady=25)

    login.mainloop()

def main_app():
    # --- Phần app chính của bạn ---

    # Tạo thư mục lưu kết quả
    os.makedirs("outputs", exist_ok=True)

    # Tải mô hình YOLO đã huấn luyện
    model = YOLO("model/yolov8n_pascal_voc2012_pro.pt")

    results_all_df = pd.DataFrame()
    global_vars = {"dark_mode": False, "webcam_running": False}  # dùng dict để thay đổi biến trong hàm lồng

    root = tk.Tk()
    root.title("Ứng dụng phát hiện đối tượng YOLOv8")
    root.state('zoomed')
    root.resizable(False, False)

    # Hàm dự đoán từ ảnh
    def run_detection():
        nonlocal results_all_df
        file_path = filedialog.askopenfilename()
        if file_path:
            # Mở ảnh bằng PIL
            image = Image.open(file_path)
            # Dự đoán bằng YOLO
            results = model.predict(source=file_path, save=False, conf=0.25)[0]

            # Load ảnh bằng PIL để giữ màu gốc, resize phù hợp
            img = image.resize((500, 400))
            imgtk = ImageTk.PhotoImage(img)
            panel.imgtk = imgtk
            panel.config(image=imgtk)

            # Vẽ annotation lên ảnh bằng OpenCV (ảnh BGR) rồi chuyển sang PIL (RGB)
            annotated = results.plot()  # OpenCV BGR ndarray
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_annotated = Image.fromarray(annotated_rgb).resize((500, 400))

            # Hiển thị ảnh có annotation
            imgtk_annotated = ImageTk.PhotoImage(img_annotated)
            panel.imgtk = imgtk_annotated
            panel.config(image=imgtk_annotated)

            data = []
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                name = model.names[int(cls)]
                data.append((os.path.basename(file_path), name, conf, int(x1), int(y1), int(x2), int(y2)))

            if data:
                df = pd.DataFrame(data, columns=["Ảnh", "Tên đối tượng", "Xác suất", "x1", "y1", "x2", "y2"])
                results_all_df = pd.concat([results_all_df, df], ignore_index=True)
                update_table(results_all_df.values)

    # Hàm cập nhật bảng kết quả
    def update_table(data):
        for item in tree.get_children():
            tree.delete(item)
        for row in data:
            tree.insert('', 'end', values=(row[0], row[1], f"{row[2]:.2f}", f"{row[3]},{row[4]},{row[5]},{row[6]}"))

    # Hàm lưu bảng kết quả ra file CSV
    def save_csv():
        if not results_all_df.empty:
            results_all_df.to_csv('outputs/results.csv', index=False)
            messagebox.showinfo("Đã lưu", "✅ Kết quả đã được lưu vào outputs/results.csv")

    # Hàm lưu ảnh đang hiển thị
    def save_image():
        path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                            filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")])
        if path and hasattr(panel, 'imgtk'):
            panel.imgtk._PhotoImage__photo.write(path, format='jpeg')
            messagebox.showinfo("Đã lưu", f"✅ Ảnh đã lưu vào: {path}")

    # Hàm bật/tắt dark mode
    def toggle_dark_mode():
        global_vars["dark_mode"] = not global_vars["dark_mode"]
        dark_mode = global_vars["dark_mode"]
        bg = "#2E2E2E" if dark_mode else "#FFFFFF"
        fg = "#FFFFFF" if dark_mode else "#000000"
        root.configure(bg=bg)
        for widget in root.winfo_children():
            if isinstance(widget, (tk.LabelFrame, tk.Frame, tk.Label)):
                widget.configure(bg=bg, fg=fg)
            for child in widget.winfo_children():
                try:
                    child.configure(bg=bg, fg=fg)
                except:
                    pass

    # Hàm chạy webcam
    def run_webcam():
        global_vars["webcam_running"] = True
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and global_vars["webcam_running"]:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, save=False, conf=0.25)[0]

            annotated = results.plot()  # BGR image
            # Chuyển đổi sang RGB để hiển thị trên Tkinter
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(annotated_rgb)
            img = img.resize((500, 400))
            imgtk = ImageTk.PhotoImage(img)
            panel.imgtk = imgtk
            panel.config(image=imgtk)

            data = []
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                name = model.names[int(cls)]
                data.append(("Webcam", name, conf, int(x1), int(y1), int(x2), int(y2)))

            if data:
                df = pd.DataFrame(data, columns=["Ảnh", "Tên đối tượng", "Xác suất", "x1", "y1", "x2", "y2"])
                nonlocal results_all_df
                results_all_df = pd.concat([results_all_df, df], ignore_index=True)
                update_table(results_all_df.values)

            root.update()
        cap.release()

    def stop_webcam():
        global_vars["webcam_running"] = False

    # GUI chính
    frame_controls = tk.LabelFrame(root, text="⚙️ Điều khiển", padx=5, pady=5)
    frame_controls.pack(pady=10, fill='x')

    btns = [
        ("📂 Chọn ảnh", run_detection),
        ("📷 Mở webcam", lambda: threading.Thread(target=run_webcam, daemon=True).start()),
        ("❌ Tắt webcam", stop_webcam),
        ("✔️ Lưu csv", save_csv),
        ("👎 Xóa kết quả", lambda: tree.delete(*tree.get_children())),
        ("🌗 Dark Mode", toggle_dark_mode)
    ]
    for i, (text, cmd) in enumerate(btns):
        ttk.Button(frame_controls, text=text, command=cmd).grid(row=0, column=i, padx=5)

    frame_display = tk.LabelFrame(root, text="💾 Ảnh/Webcam đầu vào và dự đoán", padx=5, pady=5, width=520, height=440)
    frame_display.pack(pady=10, fill='x')
    frame_display.pack_propagate(False)

    panel = tk.Label(frame_display, width=500, height=400)
    panel.pack()

    frame_table = tk.LabelFrame(root, text="📊 Kết quả phát hiện", padx=5, pady=5)
    frame_table.pack(pady=10, fill='both', expand=True)

    cols = ("Ảnh/Webcam", "Tên đối tượng", "Xác suất", "Vị trí")
    tree = ttk.Treeview(frame_table, columns=cols, show='headings')
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, anchor='center')
    tree.pack(fill='both', expand=True)

    root.mainloop()

if __name__ == "__main__":
    login_window()
