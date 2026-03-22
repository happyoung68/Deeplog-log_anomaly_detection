import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("测试窗口")
root.geometry("300x200")

label = ttk.Label(root, text="Hello, Tkinter!")
label.pack(pady=20)

button = ttk.Button(root, text="关闭", command=root.quit)
button.pack(pady=10)

root.mainloop()
