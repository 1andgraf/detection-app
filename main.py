import cv2
from ultralytics import YOLO
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from collections import Counter
import customtkinter as ctk

model = YOLO("yolov8n.pt")
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detection app")
        self.root.geometry("1200x750")
        root.minsize(1050, 700)
        self.root.configure(bg="#2b2b2b")

        self.cap = None
        self.video_path = None
        self.playing = False
        self.video_length = 0
        self.current_frame = 0

        self.setup_ui()

    def setup_ui(self):
        main_frame = Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill=BOTH, expand=True)

        # Left column - detected objects info
        info_frame = Frame(main_frame, width=200, bg="#1e1e1e")
        info_frame.pack(side=LEFT, fill=Y, padx=(0,0))
        Label(info_frame, text="Detected Objects", bg="#1e1e1e", fg="white", font=("Arial", 14)).pack(pady=(10, 5))
        self.info_text = Text(info_frame, bg="#2e2e2e", fg="white", height=40, width=30)
        self.info_text.pack(padx=10, pady=10, fill=BOTH, expand=True)

        # Center - video display
        center_frame = Frame(main_frame, bg="black", width=600, height=600)
        center_frame.pack(side=LEFT, expand=True, fill=BOTH)
        center_frame.pack_propagate(False)
        self.video_label = Label(center_frame, bg="black")
        self.video_label.pack(expand=True)

        # Media controls below video
        controls_frame = Frame(center_frame, bg="#2b2b2b")
        controls_frame.pack(fill=X, pady=10)

        self.play_button = ctk.CTkButton(controls_frame, text="▶︎", width=30, command=self.toggle_play)
        self.play_button.pack(side=LEFT, padx=10, pady=5)
        self.stop_button = ctk.CTkButton(controls_frame, text="◼︎", width=30, command=self.stop_video)
        self.stop_button.pack(side=LEFT, padx=0, pady=5)

        self.slider = ttk.Scale(controls_frame, from_=0, to=100, orient="horizontal", length=400)
        self.slider.pack(side=LEFT, padx=10, fill=X, expand=True)
        self.slider.bind("<Button-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        # Right column - detect buttons
        control_frame = Frame(main_frame, width=250, bg="#1e1e1e")
        control_frame.pack(side=RIGHT, fill=Y, padx=(0,0))
        Label(control_frame, text="Actions", bg="#1e1e1e", fg="white", font=("Arial", 14)).pack(pady=10)

        ctk.CTkButton(control_frame, text="Import Image", width=150, height=30, command=self.detect_image).pack(pady=5, padx=20)
        ctk.CTkButton(control_frame, text="Import Video", width=150, height=30, command=self.open_video).pack(pady=5, padx=20)
        
        # Inside your setup_ui() in YOLOApp, add at the bottom of right panel
        self.export_button = ctk.CTkButton(control_frame, text="Export", width=150, height=30, command=self.export_file)
        self.export_button.pack(side=BOTTOM, pady=20, padx=20)
        self.export_button.pack_forget()  # hide by default

    def update_info(self, names):
        self.info_text.delete(1.0, END)
        counter = Counter(names)
        for name, count in counter.items():
            self.info_text.insert(END, f"{name}: {count}\n")

    def show_frame(self, frame):
        h, w, _ = frame.shape
        scale = min(600 / w, 600 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def detect_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        img = cv2.imread(path)
        results = model(img)
        annotated = results[0].plot()
        names = [model.names[int(c)] for c in results[0].boxes.cls]
        self.update_info(names)
        self.show_frame(annotated)
        self.last_image = annotated
        self.export_button.pack(side=BOTTOM, pady=20, padx=20)

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.video_path = path
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.playing = False
        self.slider.set(0)
        self.show_frame_at(0)
        self.export_button.pack(side=BOTTOM, pady=20, padx=20)

    def show_frame_at(self, frame_number):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            results = model(frame)
            annotated = results[0].plot()
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            self.update_info(names)
            self.show_frame(annotated)

    def toggle_play(self):
        if not self.cap:
            return
        if self.playing:
            self.playing = False
            self.play_button.configure(text="▶")
        else:
            self.playing = True
            self.play_button.configure(text="⏸")
            self.update_video()

    def stop_video(self):
        self.playing = False
        self.play_button.configure(text="▶")
        self.current_frame = 0
        self.slider.set(0)
        self.show_frame_at(0)

    def update_video(self):
        if not self.playing or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return
        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        results = model(frame)
        annotated = results[0].plot()
        names = [model.names[int(c)] for c in results[0].boxes.cls]
        self.update_info(names)
        self.show_frame(annotated)
        if self.video_length > 0:
            self.slider.set(self.current_frame / self.video_length * 100)
        self.root.after(30, self.update_video)

    def slider_seek(self, val):
        if not self.cap or self.video_length == 0:
            return
        frame_number = int(float(val) / 100 * self.video_length)
        self.current_frame = frame_number
        self.show_frame_at(frame_number)

    def on_slider_press(self, event):
        self.was_playing = self.playing
        self.toggle_play() if self.playing else None  # pause while dragging

    def on_slider_release(self, event):
        if not self.cap or self.video_length == 0:
            return
        frame_number = int(self.slider.get() / 100 * self.video_length)
        self.show_frame_at(frame_number)
        if self.was_playing:
            self.toggle_play()  # resume if was playing
            
    def export_file(self):
        if not self.video_path and not hasattr(self, 'last_image'):
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".mp4" if self.video_path else ".png",
                                                filetypes=[("MP4 Video", "*.mp4"), ("PNG Image", "*.png")])
        if not save_path:
            return

        if self.video_path:  # export video
            cap = cv2.VideoCapture(self.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                annotated = results[0].plot()
                out.write(annotated)
            cap.release()
            out.release()

        else:  # export image
            cv2.imwrite(save_path, self.last_image)

root = Tk()
app = YOLOApp(root)
root.mainloop()