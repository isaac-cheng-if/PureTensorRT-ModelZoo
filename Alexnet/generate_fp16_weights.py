#!/usr/bin/env python3
"""
生成FP16权重文件的测试脚本
"""
import sys
import os

def main():
    # 检查输入文件是否存在
    pth_file = "alexnet-owt-7be5be79.pth"
    if not os.path.exists(pth_file):
        print(f"错误: 未找到权重文件 {pth_file}")
        print("请确保权重文件存在")
        return 1

    print(f"找到权重文件: {pth_file}")

    # 生成FP16权重
    output_file = "alexnet-tensorrt-fp16.wts"
    cmd = f"python3 convert_for_tensorrt.py {pth_file} -o {output_file}"

    print(f"执行命令: {cmd}")
    result = os.system(cmd)

    if result == 0:
        print(f"\n✅ FP16权重文件生成成功: {output_file}")
        print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        return 0
    else:
        print(f"\n❌ FP16权重文件生成失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())