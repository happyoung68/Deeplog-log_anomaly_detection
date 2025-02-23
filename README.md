# Deeplog-log_anomaly_detection
日志异常检测，Used for log anomaly detection, including log preprocessing, training, prediction, and output results.
## Introduction
Used for log anomaly detection, including log processing, training, prediction, and output results.
This work is developed on the basis of <https://github.com/d0ng1ee/logdeep>, and use [Drain](https://github.com/logpai/logparser) for log parsing.  
## Major features  
- Used for producing environment.  
- Including complete process.  
- Outputing anomaly logs, rather than precision, recall, F1-score and so on.  
## Requirement  
- python>=3.6  
- pytorch >= 1.1.0  
## Quick start  
1.Preprocess logs  
  ```
  cd demo  
  python preprocess.py
  ```
  Then you will get parsed log file at `../result/parse_result`, and `length of event_id_map` is the counts of log templates,  `../data/demo_input.csv` is the file where the EventId has been mapped to numbers starting from 1
