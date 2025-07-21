# === gui_launcher.py ===
import tkinter as tk
import threading
from camera_attendance import run_attendance_system  # ✅ FIXED
import webbrowser

def start_attendance():
    t = threading.Thread(target=run_attendance_system)
    t.start()

def open_web():
    webbrowser.open("http://localhost:3000")

root = tk.Tk()
root.title("Face Attendance System")
root.geometry("400x200")
root.config(bg="#f0f0f0")

tk.Label(root, text="Face Attendance System", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=20)

tk.Button(root, text="🟢 Start Attendance", font=("Arial", 12), bg="#4CAF50", fg="white", command=start_attendance).pack(pady=10)

tk.Button(root, text="📄 View Records", font=("Arial", 12), bg="#2196F3", fg="white", command=open_web).pack(pady=10)

tk.Button(root, text="❌ Quit", font=("Arial", 12), command=root.quit, bg="#F44336", fg="white").pack(pady=10)

root.mainloop()
