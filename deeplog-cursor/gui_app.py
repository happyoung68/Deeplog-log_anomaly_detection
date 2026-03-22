#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepLog 图形操作界面：与 README 一致 — 预处理 → 训练 → 预测；可选一键运行 demo/preprocess.py。
运行（在项目根目录）: python gui_app.py
"""

import io
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 与 README.md「Quick start」一致的流程说明（便于脱离文档操作）
WORKFLOW_README_ZH = """项目完整工作流程（摘自 README）

环境要求：Python≥3.6，PyTorch≥1.1.0；预处理阶段还依赖 logparser（Drain）、openpyxl 等（见 demo/preprocess.py）。

【步骤 1】预处理日志
  命令行（在 demo 目录下）:
    cd demo
    python preprocess.py
  作用概要:
    · 从 data/dlt_folder 读取 .dlt / .gz 等（脚本内可调路径）
    · 使用本机 DLT Viewer 转为文本并合并、排序、清洗后，用 Drain 解析
    · 解析结果目录: result/parse_result/
    · 控制台中的 length of event_id_map 表示日志模板数量
    · 最终生成带数字 EventId 的 CSV: data/demo_input.csv（EventId 从 1 起编号）

  注意: preprocess.py 中 dlt_viewer 路径默认为作者本机路径，使用前请改为你的 DLT Viewer 可执行文件路径。

【步骤 2】训练模型
  命令行（在 demo 目录下）:
    python deeplog.py train
  · 使用预处理得到的 CSV（默认 ../data/demo_input.csv）
  · 模型与参数、训练日志保存在 result/deeplog/

【步骤 3】预测并输出异常
  命令行（在 demo 目录下）:
    python deeplog.py predict
  · 异常行输出为: result/anomaly_output_for_<输入文件名>.csv

【本界面】
  · 「运行预处理脚本」等价于在 demo 目录执行 python preprocess.py，日志显示在「运行日志」页签。
  · 若你已有符合格式的 CSV（含 EventId、LineId 等），可跳过步骤 1，直接在步骤 2/3 中选择文件。

调参: 可在 demo/deeplog.py 或本界面中调整 window_size、num_candidates、训练轮数等。
"""


def list_csv_in_dir(folder):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith('.csv') and os.path.isfile(os.path.join(folder, f))
    )


def infer_num_classes(csv_path):
    df = pd.read_csv(csv_path, usecols=['EventId'])
    return int(df['EventId'].max())


def ensure_dir_sep(path):
    path = os.path.abspath(path)
    if not path.endswith(os.sep):
        path = path + os.sep
    return path


class QueueStream(io.TextIOBase):
    """将 print/tqdm 输出写入队列，供界面轮询显示。"""

    def __init__(self, q, tag='log'):
        super().__init__()
        self.q = q
        self.tag = tag
        self._buf = ''

    def write(self, s):
        if not s:
            return 0
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if line.strip():
                self.q.put((self.tag, line + '\n'))
        return len(s)

    def flush(self):
        if self._buf.strip():
            self.q.put((self.tag, self._buf))
        self._buf = ''


def build_train_options(
    train_csv_path,
    save_dir,
    device,
    window_size,
    max_epoch,
    num_classes,
    batch_size=2048,
):
    from logdeep.tools.utils import seed_everything

    seed_everything(seed=1234)
    data_dir = ensure_dir_sep(os.path.dirname(train_csv_path))
    save_dir = ensure_dir_sep(save_dir)
    options = {
        'data_dir': data_dir,
        'train_csv_path': os.path.abspath(train_csv_path),
        'window_size': int(window_size),
        'device': device,
        'sample': 'sliding_window',
        'sequentials': True,
        'quantitatives': False,
        'semantics': False,
        'feature_num': 1,
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'num_classes': int(num_classes),
        'batch_size': int(batch_size),
        'accumulation_step': 1,
        'optimizer': 'adam',
        'lr': 0.001,
        'max_epoch': int(max_epoch),
        'lr_step': (300, 350),
        'lr_decay_ratio': 0.1,
        'resume_path': None,
        'model_name': 'deeplog',
        'save_dir': save_dir,
        'model_path': os.path.join(save_dir, 'deeplog_last.pth'),
        'num_candidates': 4,
    }
    if options['max_epoch'] < 6:
        options['lr_step'] = tuple(
            max(0, min(options['max_epoch'] - 1, x)) for x in options['lr_step']
        )
    return options


def build_predict_options(
    predict_csv_path,
    model_path,
    output_csv_path,
    device,
    window_size,
    num_classes,
    num_candidates,
):
    from logdeep.tools.utils import seed_everything

    seed_everything(seed=1234)
    data_dir = ensure_dir_sep(os.path.dirname(predict_csv_path))
    options = {
        'data_dir': data_dir,
        'train_csv_path': None,
        'window_size': int(window_size),
        'device': device,
        'sequentials': True,
        'quantitatives': False,
        'semantics': False,
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'num_classes': int(num_classes),
        'batch_size': 2048,
        'model_path': os.path.abspath(model_path),
        'num_candidates': int(num_candidates),
        'predict_csv_path': os.path.abspath(predict_csv_path),
        'predict_output_csv': os.path.abspath(output_csv_path),
    }
    return options


def run_train_worker(options, log_queue):
    from logdeep.models.lstm import deeplog
    from logdeep.tools.train import Trainer

    old_out, old_err = sys.stdout, sys.stderr
    stream = QueueStream(log_queue)
    sys.stdout = stream
    sys.stderr = stream
    try:
        model = deeplog(
            input_size=options['input_size'],
            hidden_size=options['hidden_size'],
            num_layers=options['num_layers'],
            num_keys=options['num_classes'],
        )
        trainer = Trainer(model, options)
        trainer.start_train()
        log_queue.put(('done_train', options['save_dir']))
    except Exception as e:
        log_queue.put(('error', str(e)))
        import traceback
        log_queue.put(('error', traceback.format_exc()))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_out
        sys.stderr = old_err


def run_preprocess_subprocess_worker(project_root, log_queue):
    """在 demo 目录下执行 preprocess.py，将标准输出写入 log_queue。"""

    demo_dir = os.path.join(project_root, 'demo')
    script = os.path.join(demo_dir, 'preprocess.py')
    if not os.path.isfile(script):
        log_queue.put(('error', '未找到 demo/preprocess.py\n'))
        log_queue.put(('done_preprocess', -1))
        return

    def pump():
        try:
            proc = subprocess.Popen(
                [sys.executable, 'preprocess.py'],
                cwd=demo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
            )
            if proc.stdout:
                for line in proc.stdout:
                    log_queue.put(('log', line))
            code = proc.wait()
            log_queue.put(('done_preprocess', code))
        except Exception as e:
            log_queue.put(('error', str(e) + '\n'))
            import traceback
            log_queue.put(('error', traceback.format_exc()))
            log_queue.put(('done_preprocess', -1))

    threading.Thread(target=pump, daemon=True).start()


def run_predict_worker(options, log_queue):
    from logdeep.models.lstm import deeplog
    from logdeep.tools.predict import Predicter

    old_out, old_err = sys.stdout, sys.stderr
    stream = QueueStream(log_queue)
    sys.stdout = stream
    sys.stderr = stream
    try:
        model = deeplog(
            input_size=options['input_size'],
            hidden_size=options['hidden_size'],
            num_layers=options['num_layers'],
            num_keys=options['num_classes'],
        )
        predicter = Predicter(model, options)
        out_path, filtered_df = predicter.predict_unsupervised()
        log_queue.put(('done_predict', (out_path, filtered_df)))
    except Exception as e:
        log_queue.put(('error', str(e)))
        import traceback
        log_queue.put(('error', traceback.format_exc()))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_out
        sys.stderr = old_err


class DeepLogGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('DeepLog 日志异常检测')
        self.geometry('920x720')
        self.minsize(800, 600)

        self.log_queue = queue.Queue()
        self._busy = False

        default_data = os.path.join(PROJECT_ROOT, 'data')
        default_result = os.path.join(PROJECT_ROOT, 'result')
        default_model_dir = os.path.join(default_result, 'deeplog')

        main = ttk.Frame(self, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        flow = ttk.LabelFrame(main, text='完整流程概览', padding=6)
        flow.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            flow,
            text='顺序：① 预处理（Drain，生成 demo_input.csv）→ ② 训练（result/deeplog）→ ③ 预测（异常 CSV）。'
                 ' 详细说明见下方「工作流程说明」页签。',
            wraplength=860,
        ).pack(anchor=tk.W)

        pre_lab = ttk.LabelFrame(main, text='步骤 1 · 预处理（等价于 cd demo 后执行 python preprocess.py）', padding=6)
        pre_lab.pack(fill=tk.X, pady=(0, 6))
        pf0 = ttk.Frame(pre_lab)
        pf0.pack(fill=tk.X)
        ttk.Label(
            pf0,
            text='从 data/dlt_folder 等读取原始日志，经 DLT Viewer 与 Drain 解析后写入 result/parse_result，并生成 data/demo_input.csv。'
                 ' 运行前请在 demo/preprocess.py 中配置本机 DLT Viewer 路径。',
            wraplength=860,
        ).pack(anchor=tk.W, pady=(0, 4))
        pf1 = ttk.Frame(pre_lab)
        pf1.pack(fill=tk.X)
        self.btn_preprocess = ttk.Button(
            pf1, text='运行预处理脚本', command=self._on_preprocess
        )
        self.btn_preprocess.pack(side=tk.LEFT)

        # --- 训练 ---
        train_lab = ttk.LabelFrame(main, text='步骤 2 · 训练', padding=6)
        train_lab.pack(fill=tk.X, pady=(0, 6))

        self.train_dir = tk.StringVar(value=default_data)
        self.train_csv_name = tk.StringVar(value='demo_input.csv')
        row = ttk.Frame(train_lab)
        row.pack(fill=tk.X)
        ttk.Label(row, text='训练文件夹:').pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.train_dir, width=56).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True
        )
        ttk.Button(row, text='浏览…', command=self._browse_train_dir).pack(
            side=tk.LEFT
        )
        ttk.Button(row, text='刷新 CSV', command=self._refresh_train_csv).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        row2 = ttk.Frame(train_lab)
        row2.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row2, text='训练 CSV:').pack(side=tk.LEFT)
        self.train_csv_combo = ttk.Combobox(
            row2, textvariable=self.train_csv_name, width=40, state='readonly'
        )
        self.train_csv_combo.pack(side=tk.LEFT, padx=4)

        self.model_save_dir = tk.StringVar(value=default_model_dir)
        row3 = ttk.Frame(train_lab)
        row3.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row3, text='模型保存目录:').pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.model_save_dir, width=56).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True
        )
        ttk.Button(row3, text='浏览…', command=self._browse_model_dir).pack(
            side=tk.LEFT
        )

        # --- 预测 ---
        pred_lab = ttk.LabelFrame(main, text='步骤 3 · 预测', padding=6)
        pred_lab.pack(fill=tk.X, pady=(0, 6))

        self.predict_dir = tk.StringVar(value=default_data)
        self.predict_csv_name = tk.StringVar(value='demo_input.csv')
        pr = ttk.Frame(pred_lab)
        pr.pack(fill=tk.X)
        ttk.Label(pr, text='预测文件夹:').pack(side=tk.LEFT)
        ttk.Entry(pr, textvariable=self.predict_dir, width=56).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True
        )
        ttk.Button(pr, text='浏览…', command=self._browse_predict_dir).pack(
            side=tk.LEFT
        )
        ttk.Button(pr, text='刷新 CSV', command=self._refresh_predict_csv).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        pr2 = ttk.Frame(pred_lab)
        pr2.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(pr2, text='预测 CSV:').pack(side=tk.LEFT)
        self.predict_csv_combo = ttk.Combobox(
            pr2, textvariable=self.predict_csv_name, width=40, state='readonly'
        )
        self.predict_csv_combo.pack(side=tk.LEFT, padx=4)

        self.result_out_dir = tk.StringVar(value=default_result)
        pr3 = ttk.Frame(pred_lab)
        pr3.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(pr3, text='异常结果输出目录:').pack(side=tk.LEFT)
        ttk.Entry(pr3, textvariable=self.result_out_dir, width=52).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True
        )
        ttk.Button(pr3, text='浏览…', command=self._browse_result_dir).pack(
            side=tk.LEFT
        )

        self.model_path_var = tk.StringVar(
            value=os.path.join(default_model_dir, 'deeplog_last.pth')
        )
        pr4 = ttk.Frame(pred_lab)
        pr4.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(pr4, text='模型权重 (.pth):').pack(side=tk.LEFT)
        ttk.Entry(pr4, textvariable=self.model_path_var, width=52).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True
        )
        ttk.Button(pr4, text='浏览…', command=self._browse_model_pth).pack(
            side=tk.LEFT
        )

        # --- 参数 ---
        param = ttk.LabelFrame(main, text='参数', padding=6)
        param.pack(fill=tk.X, pady=(0, 6))

        pf = ttk.Frame(param)
        pf.pack(fill=tk.X)
        ttk.Label(pf, text='设备:').pack(side=tk.LEFT)
        self.device_var = tk.StringVar(value='cpu')
        ttk.Combobox(
            pf,
            textvariable=self.device_var,
            values=('cpu', 'cuda'),
            width=8,
            state='readonly',
        ).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(pf, text='窗口大小:').pack(side=tk.LEFT)
        self.window_size_var = tk.StringVar(value='10')
        ttk.Spinbox(pf, from_=3, to=50, textvariable=self.window_size_var, width=6).pack(
            side=tk.LEFT, padx=(4, 16)
        )

        ttk.Label(pf, text='Top-K 候选:').pack(side=tk.LEFT)
        self.num_cand_var = tk.StringVar(value='4')
        ttk.Spinbox(pf, from_=1, to=50, textvariable=self.num_cand_var, width=6).pack(
            side=tk.LEFT, padx=(4, 16)
        )

        ttk.Label(pf, text='训练轮数:').pack(side=tk.LEFT)
        self.max_epoch_var = tk.StringVar(value='370')
        ttk.Spinbox(pf, from_=1, to=2000, textvariable=self.max_epoch_var, width=6).pack(
            side=tk.LEFT, padx=4
        )

        # --- 按钮 ---
        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X, pady=(0, 6))
        self.btn_train = ttk.Button(
            btn_row, text='开始训练', command=self._on_train
        )
        self.btn_train.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_predict = ttk.Button(
            btn_row, text='开始预测', command=self._on_predict
        )
        self.btn_predict.pack(side=tk.LEFT)

        # --- 日志与异常预览 ---
        nb = ttk.Notebook(main)
        nb.pack(fill=tk.BOTH, expand=True)

        wf_tab = ttk.Frame(nb, padding=4)
        nb.add(wf_tab, text='工作流程说明')
        self.wf_text = tk.Text(wf_tab, height=14, wrap=tk.WORD, state=tk.DISABLED)
        wf_scroll = ttk.Scrollbar(wf_tab, command=self.wf_text.yview)
        self.wf_text.configure(yscrollcommand=wf_scroll.set)
        self.wf_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        wf_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.wf_text.configure(state=tk.NORMAL)
        self.wf_text.insert('1.0', WORKFLOW_README_ZH)
        self.wf_text.configure(state=tk.DISABLED)

        log_tab = ttk.Frame(nb, padding=4)
        nb.add(log_tab, text='运行日志')
        self.log_text = tk.Text(log_tab, height=14, wrap=tk.WORD, state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_tab, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ana_tab = ttk.Frame(nb, padding=4)
        nb.add(ana_tab, text='异常日志摘要')
        self.anomaly_text = tk.Text(ana_tab, height=14, wrap=tk.WORD, state=tk.DISABLED)
        ana_scroll = ttk.Scrollbar(ana_tab, command=self.anomaly_text.yview)
        self.anomaly_text.configure(yscrollcommand=ana_scroll.set)
        self.anomaly_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ana_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_train_csv()
        self._refresh_predict_csv()
        self.after(200, self._poll_queue)

    def _append_log(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _set_busy(self, busy):
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_preprocess.configure(state=state)
        self.btn_train.configure(state=state)
        self.btn_predict.configure(state=state)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == 'log':
                    self._append_log(self.log_text, payload)
                elif kind == 'error':
                    self._append_log(self.log_text, '[错误] ' + payload + '\n')
                elif kind == 'done_train':
                    self._append_log(
                        self.log_text,
                        '\n训练结束。模型目录: ' + str(payload) + '\n',
                    )
                    self._set_busy(False)
                    messagebox.showinfo('训练', '训练已完成。')
                elif kind == 'done_predict':
                    out_path, df = payload
                    self._append_log(
                        self.log_text,
                        '\n预测结束。输出文件: ' + out_path + '\n',
                    )
                    self._show_anomaly_summary(out_path, df)
                    self._set_busy(False)
                    messagebox.showinfo('预测', '预测已完成，异常结果已保存。')
                elif kind == 'done_preprocess':
                    self._set_busy(False)
                    code = payload
                    self._refresh_train_csv()
                    self._refresh_predict_csv()
                    if code == 0:
                        messagebox.showinfo(
                            '预处理',
                            '预处理脚本已结束（退出码 0）。'
                            ' 请确认已生成 data/demo_input.csv，必要时在步骤 2 中点击「刷新 CSV」。',
                        )
                    else:
                        messagebox.showerror(
                            '预处理',
                            '预处理脚本异常结束（退出码 %s）。请查看「运行日志」中的输出。' % code,
                        )
        except queue.Empty:
            pass
        self.after(200, self._poll_queue)

    def _show_anomaly_summary(self, out_path, df):
        self.anomaly_text.configure(state=tk.NORMAL)
        self.anomaly_text.delete('1.0', tk.END)
        n = len(df)
        lines = [
            f'输出文件: {out_path}',
            f'异常行数: {n}',
            '',
        ]
        if n == 0:
            lines.append('未检测到异常行（或输出为空）。')
        else:
            preview_cols = [
                c for c in ('LineId', 'Date', 'Time', 'Content', 'EventId', 'EventTemplate')
                if c in df.columns
            ]
            if not preview_cols:
                preview_cols = list(df.columns)[:6]
            sub = df[preview_cols].head(80)
            lines.append('预览（最多 80 行）:')
            lines.append(sub.to_string(index=False))
        self.anomaly_text.insert('1.0', '\n'.join(lines))
        self.anomaly_text.configure(state=tk.DISABLED)

    def _browse_train_dir(self):
        p = filedialog.askdirectory(initialdir=self.train_dir.get() or PROJECT_ROOT)
        if p:
            self.train_dir.set(p)
            self._refresh_train_csv()

    def _browse_predict_dir(self):
        p = filedialog.askdirectory(initialdir=self.predict_dir.get() or PROJECT_ROOT)
        if p:
            self.predict_dir.set(p)
            self._refresh_predict_csv()

    def _browse_model_dir(self):
        p = filedialog.askdirectory(
            initialdir=self.model_save_dir.get() or PROJECT_ROOT
        )
        if p:
            self.model_save_dir.set(p)
            self.model_path_var.set(os.path.join(p, 'deeplog_last.pth'))

    def _browse_result_dir(self):
        p = filedialog.askdirectory(
            initialdir=self.result_out_dir.get() or PROJECT_ROOT
        )
        if p:
            self.result_out_dir.set(p)

    def _browse_model_pth(self):
        p = filedialog.askopenfilename(
            title='选择模型权重',
            initialdir=os.path.dirname(self.model_path_var.get()) or PROJECT_ROOT,
            filetypes=[('PyTorch', '*.pth'), ('全部', '*.*')],
        )
        if p:
            self.model_path_var.set(p)

    def _refresh_train_csv(self):
        folder = self.train_dir.get().strip()
        names = list_csv_in_dir(folder)
        self.train_csv_combo['values'] = names
        if names:
            cur = self.train_csv_name.get()
            if cur not in names:
                self.train_csv_name.set(names[0])
        else:
            self.train_csv_name.set('')

    def _refresh_predict_csv(self):
        folder = self.predict_dir.get().strip()
        names = list_csv_in_dir(folder)
        self.predict_csv_combo['values'] = names
        if names:
            cur = self.predict_csv_name.get()
            if cur not in names:
                self.predict_csv_name.set(names[0])
        else:
            self.predict_csv_name.set('')

    def _train_csv_full(self):
        folder = self.train_dir.get().strip()
        name = self.train_csv_name.get().strip()
        if not folder or not name:
            return None
        return os.path.join(folder, name)

    def _predict_csv_full(self):
        folder = self.predict_dir.get().strip()
        name = self.predict_csv_name.get().strip()
        if not folder or not name:
            return None
        return os.path.join(folder, name)

    def _on_preprocess(self):
        if self._busy:
            return
        ok = messagebox.askokcancel(
            '预处理',
            '将在目录 demo/ 下执行：python preprocess.py\n\n'
            '请确认已在 demo/preprocess.py 中设置本机 DLT Viewer 路径，'
            '且已在 data/dlt_folder（或脚本内路径）放入待处理日志。\n\n'
            '若你不需要 DLT 流程、已有预处理好的 CSV，可点「取消」，直接用步骤 2/3。',
            icon=messagebox.WARNING,
        )
        if not ok:
            return
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self._append_log(
            self.log_text,
            '开始执行预处理: demo/preprocess.py（工作目录 demo/）\n',
        )
        self._set_busy(True)
        run_preprocess_subprocess_worker(PROJECT_ROOT, self.log_queue)

    def _on_train(self):
        if self._busy:
            return
        path = self._train_csv_full()
        if not path or not os.path.isfile(path):
            messagebox.showwarning('训练', '请选择有效的训练 CSV 文件。')
            return
        try:
            num_cls = infer_num_classes(path)
        except Exception as e:
            messagebox.showerror('训练', '无法读取 EventId 列: ' + str(e))
            return
        save_d = self.model_save_dir.get().strip()
        if not save_d:
            messagebox.showwarning('训练', '请指定模型保存目录。')
            return
        device = self.device_var.get().strip()
        if device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    messagebox.showwarning('训练', 'CUDA 不可用，将使用 CPU。')
                    device = 'cpu'
                    self.device_var.set('cpu')
            except ImportError:
                device = 'cpu'
                self.device_var.set('cpu')

        try:
            ws = int(self.window_size_var.get())
            me = int(self.max_epoch_var.get())
        except ValueError:
            messagebox.showerror('训练', '窗口大小与训练轮数必须为整数。')
            return

        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self._append_log(self.log_text, f'开始训练…\n数据: {path}\n类别数(num_classes): {num_cls}\n')

        opts = build_train_options(
            train_csv_path=path,
            save_dir=save_d,
            device=device,
            window_size=ws,
            max_epoch=me,
            num_classes=num_cls,
        )

        self._set_busy(True)
        threading.Thread(
            target=run_train_worker, args=(opts, self.log_queue), daemon=True
        ).start()

    def _on_predict(self):
        if self._busy:
            return
        path = self._predict_csv_full()
        if not path or not os.path.isfile(path):
            messagebox.showwarning('预测', '请选择有效的预测 CSV 文件。')
            return
        mp = self.model_path_var.get().strip()
        if not mp or not os.path.isfile(mp):
            messagebox.showwarning('预测', '请选择有效的模型 .pth 文件。')
            return
        out_dir = self.result_out_dir.get().strip()
        if not out_dir:
            messagebox.showwarning('预测', '请指定异常结果输出目录。')
            return
        try:
            num_cls = infer_num_classes(path)
        except Exception as e:
            messagebox.showerror('预测', '无法读取 EventId 列: ' + str(e))
            return

        base = os.path.basename(path)
        out_csv = os.path.join(out_dir, 'anomaly_output_for_' + base)

        try:
            ws = int(self.window_size_var.get())
            nc = int(self.num_cand_var.get())
        except ValueError:
            messagebox.showerror('预测', '窗口大小与 Top-K 必须为整数。')
            return

        device = self.device_var.get().strip()
        if device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    messagebox.showwarning('预测', 'CUDA 不可用，将使用 CPU。')
                    device = 'cpu'
                    self.device_var.set('cpu')
            except ImportError:
                device = 'cpu'
                self.device_var.set('cpu')

        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self._append_log(
            self.log_text,
            f'开始预测…\n数据: {path}\n模型: {mp}\n输出: {out_csv}\n类别数: {num_cls}\n',
        )

        opts = build_predict_options(
            predict_csv_path=path,
            model_path=mp,
            output_csv_path=out_csv,
            device=device,
            window_size=ws,
            num_classes=num_cls,
            num_candidates=nc,
        )

        self._set_busy(True)
        threading.Thread(
            target=run_predict_worker, args=(opts, self.log_queue), daemon=True
        ).start()


def main():
    app = DeepLogGui()
    app.mainloop()


if __name__ == '__main__':
    main()
