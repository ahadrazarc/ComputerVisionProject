import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from detector import VideoDetector

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Video Object Detection Tool")
        self.root.geometry("550x500")
        self.root.resizable(False, False)

        # Set custom window icon
        try:
            self.root.iconbitmap("Computer Vision.ico")  # Replace with your .ico file
        except Exception as e:
            print("Window icon load failed:", e)

        self.model_path = None
        self.video_path = None

        self.setup_ui()

    def setup_ui(self):
        # Load logo (optional image file for branding)
        try:
            logo_image = Image.open("logo.png")
            logo_image = logo_image.resize((110, 110), Image.ANTIALIAS)
            logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = tk.Label(self.root, image=logo_photo, bg="white")
            logo_label.image = logo_photo
            logo_label.pack(pady=(15, 5))
        except Exception as e:
            print("Logo load error:", e)

        # Header label
        header = tk.Label(self.root,
                          text="YOLOv8 Video Object Detection Tool",
                          font=("Helvetica", 16, "bold"),
                          fg="#2c3e50")
        header.pack(pady=(0, 15))

        # Upload Model Button
        tk.Button(self.root,
                  text="📁 Upload YOLOv8 Model (.pt)",
                  font=("Helvetica", 12),
                  width=40,
                  command=self.upload_model,
                  bg="#e0e0e0").pack(pady=10)

        # Upload Video Button
        tk.Button(self.root,
                  text="🎞️ Upload Video File (.mp4)",
                  font=("Helvetica", 12),
                  width=40,
                  command=self.upload_video,
                  bg="#e0e0e0").pack(pady=10)

        # Run Detection Button
        tk.Button(self.root,
                  text="▶ Run Detection",
                  font=("Helvetica", 13, "bold"),
                  width=30,
                  command=self.run_detection,
                  bg="#4CAF50",
                  fg="white").pack(pady=25)

        # Status Label (multi-line support)
        self.status_text = tk.StringVar()
        self.status_text.set("Please upload model and video files.")
        self.status_label = tk.Label(self.root,
                                     textvariable=self.status_text,
                                     font=("Helvetica", 10),
                                     fg="gray",
                                     justify="left",
                                     wraplength=500)
        self.status_label.pack(pady=(10, 5))

    def upload_model(self):
        file_path = filedialog.askopenfilename(
            title="Select YOLOv8 .pt Model File",
            filetypes=[("PyTorch Model", "*.pt")]
        )
        if file_path:
            self.model_path = file_path
            self.status_text.set(f"✅ Model loaded:\n{os.path.basename(file_path)}")

    def upload_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if file_path:
            self.video_path = file_path
            self.status_text.set(f"🎞️ Video loaded:\n{os.path.basename(file_path)}")

    def run_detection(self):
        if not self.model_path:
            messagebox.showwarning("Model Missing", "Please upload a YOLOv8 model file (.pt).")
            return

        if not self.video_path:
            messagebox.showwarning("Video Missing", "Please upload a video file (.mp4).")
            return

        try:
            self.status_text.set("🔄 Running detection...\nPlease wait, processing video.")
            self.root.update_idletasks()  # Force UI update

            # Run detection
            detector = VideoDetector(self.model_path)
            output_path = detector.process(self.video_path)

            self.status_text.set(f"✅ Detection completed.\nOutput saved as: {os.path.basename(output_path)}")
            messagebox.showinfo("Success", f"Video processed successfully!\nSaved as:\n{output_path}")

        except Exception as e:
            self.status_text.set("❌ Error during detection.")
            messagebox.showerror("Error", str(e))


# Start the App
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()
