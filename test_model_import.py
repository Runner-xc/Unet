#!/usr/bin/env python3
"""
测试模型导入是否正常
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_imports():
    """测试模型导入"""
    try:
        print("🔧 测试模型导入...")
        
        # 测试基础导入
        import torch
        print("✅ PyTorch导入成功")
        
        # 测试我们修复的模型导入
        print("🔄 测试模型模块导入...")
        from src.models import *
        print("✅ 所有模型导入成功！")
        
        print("\n🎉 导入路径修复成功！")
        print("现在你可以正常运行:")
        print("  cd examples")
        print("  python app.py")
        print("\n注意：如果还有其他错误，可能是缺少依赖包，请运行:")
        print("  pip install gradio torchvision pillow numpy")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 UNet模型导入测试")
    print("=" * 60)
    
    success = test_model_imports()
    
    print("=" * 60)
    if success:
        print("✅ 导入路径修复成功！")
    else:
        print("❌ 仍有导入问题需要解决")
    
    sys.exit(0 if success else 1)