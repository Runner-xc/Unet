# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/11/5 16:50:41
*      AUTHOR: @Runner and @Mr.Robot
*      DESCRIPTION: 运行终端命令
=================================================
'''

import subprocess
import time

def run_shell_command(command):
    """
    运行shell命令
    """
    try:
        # print(f"正在运行shell命令: {command}")
        results = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, err = results.communicate()
        if results.returncode == 0:
            print(f"命令执行成功！{output}")
        else:
            print(f"命令执行失败！, 请查看错误信息：{err}")
        return results.returncode == 0
    except Exception as e:
        print(f"执行命令时发生异常：{e}")
        return False

def if_port_is_in_use(PORT):
    """
    检查端口是否被占用
    """
    while True:
        command = f"lsof -i :{PORT}"
        a = run_shell_command(command)
        if a is True:
            print(f"端口:{PORT}被占用")
            PORT += 1
        else:
            print(f"端口:{PORT}未被占用")
            return PORT
        

def run_tensorboard(log_path, PORT=6006, HOST='0.0.0.0'):
    """
    启动TensorBoard面板
    确保是在cv环境下运行，否则无法启动
    """      
    print(f"😃 正在启动 tensorboard面板...\nlog_path: {log_path}")
    PORT = if_port_is_in_use(PORT)
    command = f"nohup python3 -m tensorboard.main --logdir='{log_path}' --port={PORT} --host={HOST} > /dev/null 2>&1 &"
    if not run_shell_command(command):
        print(f"TensorBoard 启动失败！")
    else:
        time.sleep(5)
        print(f"😃 TensorBoard 启动成功！\n请访问 localhost:{PORT} 查看TensorBoard面板。")

if __name__ == "__main__":
    log_path = '/root/Unet/results/logs/DL_unet/L: DiceLoss--S: CosineAnnealingLR/optim: AdamW-lr: 0.0008-wd: 1e-06/2024-10-18_09:39:07'
    run_tensorboard(log_path, PORT=6006, HOST='0.0.0.0')