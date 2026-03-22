import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import subprocess
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import shutil

class DeepLogGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepLog 日志异常检测")
        self.root.geometry("900x600")
        self.root.resizable(True, True)
        
        # 设置主题
        style = ttk.Style()
        style.theme_use('clam')
        
        # 创建选项卡
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 训练选项卡
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="训练")
        
        # 预测选项卡
        self.predict_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_tab, text="预测")
        
        # 初始化训练选项卡
        self.init_train_tab()
        
        # 初始化预测选项卡
        self.init_predict_tab()
        
        # 训练线程
        self.train_thread = None
        self.train_running = False
        
        # 预测线程
        self.predict_thread = None
        self.predict_running = False
    
    def init_train_tab(self):
        # 创建训练选项卡的布局
        main_frame = ttk.Frame(self.train_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 文件夹选择区域
        folder_frame = ttk.LabelFrame(main_frame, text="数据文件夹选择")
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.folder_path = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_path, width=60)
        folder_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(folder_frame, text="浏览", command=self.browse_folder)
        browse_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 训练参数区域
        params_frame = ttk.LabelFrame(main_frame, text="训练参数")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 窗口大小
        window_frame = ttk.Frame(params_frame)
        window_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(window_frame, text="窗口大小:", width=15).pack(side=tk.LEFT, padx=5)
        self.window_size = tk.StringVar(value="10")
        ttk.Entry(window_frame, textvariable=self.window_size, width=10).pack(side=tk.LEFT, padx=5)
        
        # 隐藏层大小
        hidden_frame = ttk.Frame(params_frame)
        hidden_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(hidden_frame, text="隐藏层大小:", width=15).pack(side=tk.LEFT, padx=5)
        self.hidden_size = tk.StringVar(value="64")
        ttk.Entry(hidden_frame, textvariable=self.hidden_size, width=10).pack(side=tk.LEFT, padx=5)
        
        # 学习率
        lr_frame = ttk.Frame(params_frame)
        lr_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(lr_frame, text="学习率:", width=15).pack(side=tk.LEFT, padx=5)
        self.lr = tk.StringVar(value="0.001")
        ttk.Entry(lr_frame, textvariable=self.lr, width=10).pack(side=tk.LEFT, padx=5)
        
        # 训练控制区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.train_btn = ttk.Button(control_frame, text="开始训练", command=self.start_train, style="Accent.TButton")
        self.train_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="停止训练", command=self.stop_train, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 训练状态区域
        status_frame = ttk.LabelFrame(main_frame, text="训练状态")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 训练日志
        log_frame = ttk.Frame(status_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.log_text.yview)
        
        # 训练进度条
        progress_frame = ttk.Frame(status_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="准备就绪")
        self.progress_label.pack(side=tk.RIGHT, padx=5)
    
    def init_predict_tab(self):
        # 创建预测选项卡的布局
        main_frame = ttk.Frame(self.predict_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 文件夹选择区域
        folder_frame = ttk.LabelFrame(main_frame, text="预测数据文件夹选择")
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.predict_folder_path = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.predict_folder_path, width=60)
        folder_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(folder_frame, text="浏览", command=self.browse_predict_folder)
        browse_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 预测控制区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.predict_btn = ttk.Button(control_frame, text="开始预测", command=self.start_predict, style="Accent.TButton")
        self.predict_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.export_btn = ttk.Button(control_frame, text="导出结果", command=self.export_result, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 预测结果区域
        result_frame = ttk.LabelFrame(main_frame, text="预测结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 结果日志
        log_frame = ttk.Frame(result_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(log_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.result_text.yview)
        
        # 结果统计
        stats_frame = ttk.Frame(result_frame)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.anomaly_count = tk.StringVar(value="0")
        ttk.Label(stats_frame, text="异常日志数量:", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Label(stats_frame, textvariable=self.anomaly_count, width=10).pack(side=tk.LEFT, padx=5)
        
        # 结果可视化
        vis_frame = ttk.LabelFrame(result_frame, text="结果可视化")
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path.set(folder_selected)
    
    def browse_predict_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.predict_folder_path.set(folder_selected)
    
    def start_train(self):
        # 检查文件夹是否选择
        if not self.folder_path.get():
            messagebox.showerror("错误", "请选择训练数据文件夹")
            return
        
        # 检查文件夹是否存在
        if not os.path.exists(self.folder_path.get()):
            messagebox.showerror("错误", "选择的文件夹不存在")
            return
        
        # 禁用训练按钮，启用停止按钮
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # 清空日志
        self.log_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.progress_label.config(text="开始训练...")
        
        # 启动训练线程
        self.train_running = True
        self.train_thread = threading.Thread(target=self.run_train)
        self.train_thread.daemon = True
        self.train_thread.start()
    
    def run_train(self):
        try:
            # 复制数据到data目录
            data_dir = os.path.join(os.path.dirname(__file__), 'data', 'dlt_folder')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            # 清空data_dir
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # 复制文件
            for file in os.listdir(self.folder_path.get()):
                if file.endswith('.dlt') or file.endswith('.gz'):
                    src_path = os.path.join(self.folder_path.get(), file)
                    dst_path = os.path.join(data_dir, file)
                    os.copy(src_path, dst_path)
                    self.log_text.insert(tk.END, f"复制文件: {file}\n")
                    self.log_text.see(tk.END)
            
            # 预处理数据
            self.log_text.insert(tk.END, "\n开始预处理数据...\n")
            self.log_text.see(tk.END)
            
            preprocess_script = os.path.join(os.path.dirname(__file__), 'demo', 'preprocess.py')
            process = subprocess.Popen(
                [sys.executable, preprocess_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), 'demo')
            )
            
            for line in iter(process.stdout.readline, ''):
                if not self.train_running:
                    process.terminate()
                    break
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
                self.root.update()
            
            process.wait()
            
            if process.returncode != 0:
                self.log_text.insert(tk.END, f"预处理失败，返回码: {process.returncode}\n")
                self.log_text.see(tk.END)
                self.train_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.progress_label.config(text="预处理失败")
                return
            
            self.log_text.insert(tk.END, "\n预处理完成，开始训练模型...\n")
            self.log_text.see(tk.END)
            
            # 训练模型
            deeplog_script = os.path.join(os.path.dirname(__file__), 'demo', 'deeplog.py')
            process = subprocess.Popen(
                [sys.executable, deeplog_script, 'train'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), 'demo')
            )
            
            # 模拟进度
            epoch = 0
            max_epoch = 370
            
            for line in iter(process.stdout.readline, ''):
                if not self.train_running:
                    process.terminate()
                    break
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
                
                # 检查是否有epoch信息
                if 'Starting epoch:' in line:
                    epoch += 1
                    progress = (epoch / max_epoch) * 100
                    self.progress_var.set(progress)
                    self.progress_label.config(text=f"训练中... Epoch {epoch}/{max_epoch}")
                
                self.root.update()
            
            process.wait()
            
            if process.returncode != 0:
                self.log_text.insert(tk.END, f"训练失败，返回码: {process.returncode}\n")
                self.log_text.see(tk.END)
                self.progress_label.config(text="训练失败")
            else:
                self.log_text.insert(tk.END, "\n训练完成！\n")
                self.log_text.see(tk.END)
                self.progress_var.set(100)
                self.progress_label.config(text="训练完成")
                
        except Exception as e:
            self.log_text.insert(tk.END, f"错误: {str(e)}\n")
            self.log_text.see(tk.END)
            self.progress_label.config(text="发生错误")
        finally:
            self.train_running = False
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def stop_train(self):
        self.train_running = False
        self.log_text.insert(tk.END, "\n停止训练...\n")
        self.log_text.see(tk.END)
    
    def start_predict(self):
        # 检查文件夹是否选择
        if not self.predict_folder_path.get():
            messagebox.showerror("错误", "请选择预测数据文件夹")
            return
        
        # 检查文件夹是否存在
        if not os.path.exists(self.predict_folder_path.get()):
            messagebox.showerror("错误", "选择的文件夹不存在")
            return
        
        # 禁用预测按钮
        self.predict_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        
        # 清空结果
        self.result_text.delete(1.0, tk.END)
        self.anomaly_count.set("0")
        
        # 启动预测线程
        self.predict_running = True
        self.predict_thread = threading.Thread(target=self.run_predict)
        self.predict_thread.daemon = True
        self.predict_thread.start()
    
    def run_predict(self):
        try:
            # 复制数据到data目录
            data_dir = os.path.join(os.path.dirname(__file__), 'data', 'dlt_folder')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            # 清空data_dir
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # 复制文件
            for file in os.listdir(self.predict_folder_path.get()):
                if file.endswith('.dlt') or file.endswith('.gz'):
                    src_path = os.path.join(self.predict_folder_path.get(), file)
                    dst_path = os.path.join(data_dir, file)
                    os.copy(src_path, dst_path)
                    self.result_text.insert(tk.END, f"复制文件: {file}\n")
                    self.result_text.see(tk.END)
            
            # 预处理数据
            self.result_text.insert(tk.END, "\n开始预处理数据...\n")
            self.result_text.see(tk.END)
            
            preprocess_script = os.path.join(os.path.dirname(__file__), 'demo', 'preprocess.py')
            process = subprocess.Popen(
                [sys.executable, preprocess_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), 'demo')
            )
            
            for line in iter(process.stdout.readline, ''):
                if not self.predict_running:
                    process.terminate()
                    break
                self.result_text.insert(tk.END, line)
                self.result_text.see(tk.END)
                self.root.update()
            
            process.wait()
            
            if process.returncode != 0:
                self.result_text.insert(tk.END, f"预处理失败，返回码: {process.returncode}\n")
                self.result_text.see(tk.END)
                self.predict_btn.config(state=tk.NORMAL)
                return
            
            self.result_text.insert(tk.END, "\n预处理完成，开始预测...\n")
            self.result_text.see(tk.END)
            
            # 预测
            deeplog_script = os.path.join(os.path.dirname(__file__), 'demo', 'deeplog.py')
            process = subprocess.Popen(
                [sys.executable, deeplog_script, 'predict'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), 'demo')
            )
            
            for line in iter(process.stdout.readline, ''):
                if not self.predict_running:
                    process.terminate()
                    break
                self.result_text.insert(tk.END, line)
                self.result_text.see(tk.END)
                self.root.update()
            
            process.wait()
            
            if process.returncode != 0:
                self.result_text.insert(tk.END, f"预测失败，返回码: {process.returncode}\n")
                self.result_text.see(tk.END)
            else:
                self.result_text.insert(tk.END, "\n预测完成！\n")
                self.result_text.see(tk.END)
                
                # 加载结果
                result_file = os.path.join(os.path.dirname(__file__), 'result', 'anomaly_output_for_demo_input.csv')
                if os.path.exists(result_file):
                    df = pd.read_csv(result_file)
                    anomaly_count = len(df)
                    self.anomaly_count.set(str(anomaly_count))
                    
                    # 可视化结果
                    self.ax.clear()
                    if anomaly_count > 0:
                        # 按小时统计异常
                        df['Date'] = pd.to_datetime(df['Date'])
                        df['Hour'] = df['Date'].dt.hour
                        hour_counts = df['Hour'].value_counts().sort_index()
                        
                        self.ax.bar(hour_counts.index, hour_counts.values)
                        self.ax.set_xlabel('小时')
                        self.ax.set_ylabel('异常数量')
                        self.ax.set_title('异常日志小时分布')
                        self.ax.set_xticks(range(24))
                    else:
                        self.ax.text(0.5, 0.5, '未发现异常', ha='center', va='center')
                    
                    self.canvas.draw()
                    
                    # 启用导出按钮
                    self.export_btn.config(state=tk.NORMAL)
                
        except Exception as e:
            self.result_text.insert(tk.END, f"错误: {str(e)}\n")
            self.result_text.see(tk.END)
        finally:
            self.predict_running = False
            self.predict_btn.config(state=tk.NORMAL)
    
    def export_result(self):
        # 导出结果文件
        result_file = os.path.join(os.path.dirname(__file__), 'result', 'anomaly_output_for_demo_input.csv')
        if not os.path.exists(result_file):
            messagebox.showerror("错误", "结果文件不存在")
            return
        
        export_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="anomaly_output.csv"
        )
        
        if export_path:
            try:
                shutil.copy(result_file, export_path)
                messagebox.showinfo("成功", f"结果已导出到: {export_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {str(e)}")

if __name__ == "__main__":
    import shutil
    root = tk.Tk()
    app = DeepLogGUI(root)
    root.mainloop()
