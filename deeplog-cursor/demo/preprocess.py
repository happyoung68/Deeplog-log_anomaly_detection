import os
import subprocess
import pandas as pd
from openpyxl import Workbook
import openpyxl
from openpyxl.styles import Font, Color, Alignment
from openpyxl import load_workbook
from datetime import datetime
import gzip
import re
import ast
import shutil
import os

from logparser.Drain import LogParser

import sys
import logging
import pandas as pd
from spellpy import spell

# logging.basicConfig(level=logging.WARNING,
#                     format='[%(asctime)s][%(levelname)s]: %(message)s')
# logger = logging.getLogger(__name__)


def convert_dlt_to_txt(input_dlt_file, output_txt_file, dlt_viewer_path):
    try:
        # 构造调用DLT Viewer的命令
        command = [dlt_viewer_path, '-c', input_dlt_file, output_txt_file]
        # 执行命令
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # 检查命令执行结果
        if result.returncode == 0:
            print(f"DLT文件 {input_dlt_file} 转换为TXT文件成功！")
        else:
            print("DLT文件转换失败：", result.stderr)
    except Exception as e:
        print("DLT文件转换失败：", e)


def decompress_folder(folder):
    # 获取文件夹中所有文件的列表
    files = os.listdir(folder)

    # 遍历每个文件
    for file in files:
        if file.endswith('.gz'):
            gzipped_file = os.path.join(folder, file)
            # 输出文件的路径，去掉.gz后缀
            output_file = os.path.join(folder, os.path.splitext(file)[0])

            # 检查输出文件是否已经存在
            if os.path.exists(output_file + ".dlt"):
                print(f"已存在解压后的文件: {output_file}")
                continue  # 如果文件已存在，则跳过解压缩

            # 解压缩.gz文件到输出文件
            with gzip.open(gzipped_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            print(f"解压缩 {gzipped_file} 到 {output_file}")


def ensure_dlt_extension(filename):
    # 检查文件名是否有后缀
    if '.' not in filename:
        new_filename = filename + '.dlt'
    else:
        # 获取文件名和后缀名
        base_name, extension = os.path.splitext(filename)

        # 如果后缀名不是 '.dlt'，则修改为 '.dlt'
        if extension != '.dlt':
            new_filename = base_name + '.dlt'
        else:
            new_filename = filename  # 后缀名已经是 '.dlt'，不需要修改

    # 如果新文件名和旧文件名相同，直接返回
    if new_filename == filename:
        return filename

    # 保存文件（这里假设使用重命名操作来模拟保存）
    try:

        os.rename(filename, new_filename)
        print(f'文件 "{filename}" 重命名为 "{new_filename}"')
        return new_filename
    except OSError as e:
        print(f'FFFError: {e}')
        return filename


folder_path = '../data/dlt_folder'

decompress_folder(folder_path)

files = os.listdir(folder_path)
input_dlt_file = [file for file in files if file.endswith('.dlt') or file.endswith('.log') or '.' not in file]
for i in range(0, len(input_dlt_file)):
    input_dlt_file[i] = ensure_dlt_extension(folder_path + "/" + input_dlt_file[i])
print(input_dlt_file)

output_txt_file = []
for i in range(0, len(input_dlt_file)):
    output_txt_file.append(input_dlt_file[i].replace(".dlt", ".txt"))
for i in range(0, len(output_txt_file)):
    with open(output_txt_file[i], 'w') as file:
        pass
print(output_txt_file)
all_output_txt_file = "AAAAAAAAALLLL__.txt"

with open(all_output_txt_file, 'w') as file:
    pass

# 指定 DLT Viewer 的完整路径
dlt_viewer_path = 'E:\AAA_Using_Tool_ZXY_Artist\DltViewer_2.17.0_Stable\DltViewer_2.17.0_Stable\dlt_viewer'

# 调用转换函数
for i in range(0, len(input_dlt_file)):
    convert_dlt_to_txt(input_dlt_file[i], output_txt_file[i], dlt_viewer_path)

# 合并TXT文件
try:
    with open(all_output_txt_file, 'w', encoding='utf-8') as outfile:
        for file_name in output_txt_file:
            with open(file_name, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    print(f"合并完成，结果保存在 {all_output_txt_file}")
except Exception as e:
    print("合并过程中出现错误:", e)

# 删除单个TXT文件
try:
    for file_path in output_txt_file:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"临时TXT文件 {file_path} 已经删除.")
        else:
            print(f"临时TXT文件 {file_path} 不存在.")
except Exception as e:
    print("删除临时TXT文件失败:", e)

# 读取文件
with open(all_output_txt_file, 'r') as file:
    lines = file.readlines()


# 定义一个函数来提取日期和时间并合并成一个 datetime 对象
def get_datetime_from_line(line):
    # 假设每一行的格式是：
    # <数值> <日期> <时间> ...
    # 例如：12 2024/10/13 20:58:42.991538 90.1859 225 CCU0 POWM CTX1 4096 log error verbose 1 Leave PreS,time rest 1993
    parts = line.split()
    date_str = parts[1]  # 获取日期部分（2024/10/13）
    time_str = parts[2]  # 获取时间部分（20:58:42.991538）

    # 将日期和时间字符串合并为一个完整的字符串 "2024/10/13 20:58:42.991538"
    datetime_str = f"{date_str} {time_str}"

    # 将合并后的字符串转换为 datetime 对象
    return datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S.%f")


# 根据日期和时间对行进行排序
sorted_lines = sorted(lines, key=get_datetime_from_line)

# 将排序后的内容写回到文件或输出
with open('sorted_file.txt', 'w') as file:
    file.writelines(sorted_lines)

print("文件已按照日期和时间排序并保存为 sorted_file.txt")

with open('sorted_file.txt', 'r+') as file:
    lines = file.readlines()  # 读取所有行

    # 遍历每一行
    for i, line in enumerate(lines):
        # 使用正则表达式删除类似 [008][#:00.0000s] 的部分
        lines[i] = re.sub(r'\[\d+\]\[#:\d+\.\d+s\]', '', line).strip() + '\n'

    # 将修改后的内容写回文件
    file.seek(0)  # 将文件指针移到文件开始
    file.writelines(lines)  # 写回修改后的所有行
    file.truncate()  # 如果文件减少了行数，确保文件被正确截断

print("文件已删除带#号的相对开机时间内容")

try:
    if os.path.exists(all_output_txt_file):
        os.remove(all_output_txt_file)
        print(f"临时TXT文件 {all_output_txt_file} 已经删除.")
    else:
        print(f"临时TXT文件 {all_output_txt_file} 不存在.")
except Exception as e:
    print("删除临时TXT文件失败:", e)


input_dir = "./"  # The input directory of log file
# log_file = 'unknow.txt'  # The input log file name
log_file = 'sorted_file.txt'
# log_format = '<Date> <Time> <Level>:<Content>' # Define log format to split message fields
log_format = '<Index> <Date> <Time> <TimeStamp> <Count> <Ecuid> <Apid> <Ctid> <Sessionid> <Type> <Subtype> <Mode> <Args> <Content>'
# Regular expression list for optional preprocessing (default: [])
regex = [
    # r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)' # IP
    # r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
    r'(?<=[^A-Za-z0-9])([0-9A-Fa-f]{2})(?=[^A-Za-z0-9])|[0-9A-Fa-f]{2}$'
]
st = 0.5  # Similarity threshold
# depth = 4  # Depth of all leaf nodes
depth = 4
parser = LogParser(log_format=log_format, outdir= '../result/parse_result', depth=depth, st=st, rex=regex)
# parser = LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st)
parser.parse(log_file)

try:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"临时TXT文件 {log_file} 已经删除.")
    else:
        print(f"临时TXT文件 {log_file} 不存在.")
except Exception as e:
    print("删除临时TXT文件失败:", e)

df = pd.read_csv('../result/parse_result/sorted_file.txt_structured.csv')
event_id_map = dict()
for i, event_id in enumerate(df['EventId'].unique(), 1):
    event_id_map[event_id] = i

print(f'length of event_id_map: {len(event_id_map)}')

df['EventId'] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e, -2) != -2 else -1)
df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# df['ParameterList'] = df['ParameterList'].apply(lambda x: [item.split(',') for item in ast.literal_eval(x)])
df['ParameterList'] = df['ParameterList'].apply(ast.literal_eval)
df['ParameterList'] = df['ParameterList'].apply(lambda x: x.append('0') or x)

df.to_csv('../data/demo_input.csv', index=False)

