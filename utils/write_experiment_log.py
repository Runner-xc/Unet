import time

def current_time():
    return time.strftime("%Y-%m-%d_%H:%M:%S")


def write_exp_logs(save_path, content):
    with open (save_path, 'w', encoding='utf-8') as f:
        f.write("# 实验日志\n")
        f.write(f"## 时间：{current_time()}\n")
    with open (save_path, 'a', encoding='utf-8') as f:
        f.write("```python\n") 
        f.write(f"{content}\n")
        f.write("```")
    return save_path