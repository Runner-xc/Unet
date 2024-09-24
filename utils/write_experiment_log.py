import time

def current_time():
    return time.strptime("%Y-%m-%d_%H:%M:%S")

def write_exp_logs(save_path, content):
    with open (save_path, 'w', default='utf-8') as f:
        f.write("# 实验日志")
        f.write(f"## 时间：{current_time}")
    with open (save_path, 'a', encoding='utf-8') as f:
        f.write(f"```{content}```")
    return save_path