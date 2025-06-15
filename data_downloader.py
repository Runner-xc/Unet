import kaggle
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, DownloadColumn
import os
import zipfile

def download_kaggle_dataset(dataset_name: str, path: Path):
    """
    下载Kaggle数据集到指定路径, 并显示下载进度条。
    注意:下载前请先配置kaggle API密钥!!!     

    :param dataset_name: Kaggle数据集的名称, 格式为 'username/dataset-name'
    :param path: 保存数据集的本地路径
    """
    # 确保目录存在
    path.mkdir(parents=True, exist_ok=True)
    
    # 认证Kaggle API
    kaggle.api.authenticate()
    
    # 创建临时ZIP文件路径
    zip_path = path / f"{dataset_name.split('/')[-1]}.zip"
    
    # 使用rich创建进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TimeRemainingColumn(),
        transient=True
    ) as progress:
        
        # 创建下载任务
        download_task = progress.add_task(
            "download", 
            filename=dataset_name.split('/')[-1], 
            total=None
        )
        
        # 使用 Kaggle API 下载数据集
        kaggle.api.dataset_download_files(dataset_name, path=str(path), unzip=False, quiet=False)
        
        # 更新进度条（由于 Kaggle API 自动处理下载，进度条无法实时更新）
        progress.update(download_task, completed=100)
        
        # 添加解压进度条
        if zip_path.exists():
            extract_task = progress.add_task("extract", filename="Extracting...", total=100)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.infolist()
                total_size = sum(m.file_size for m in members)
                extracted_size = 0
                
                for i, member in enumerate(members):
                    zip_ref.extract(member, path=str(path))
                    extracted_size += member.file_size
                    progress.update(extract_task, completed=int(extracted_size * 100 / total_size))
            
            # 删除临时ZIP文件
            zip_path.unlink()
            progress.update(extract_task, visible=False)
        
        print(f"Dataset '{dataset_name}' downloaded and extracted to {path}")

if __name__ == "__main__":
    dataset_name = "kmader/skin-cancer-mnist-ham10000"  
    download_path = Path("./datasets_kaggle") 
    # 下载数据集
    download_kaggle_dataset(dataset_name=dataset_name, path=download_path)