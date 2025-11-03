#!/usr/bin/env python3
"""
"""
import torch
import struct
import argparse
import numpy as np

def apply_weight_transforms_for_tensorrt(name, tensor):
    """
    权重在提取时直接转换为fp16，避免运行时转换开销

    Args:
        name (str): 权重名称
        tensor (torch.Tensor): 权重张量 (fp32)

    Returns:
        torch.Tensor: 转换后的权重张量 (转换为fp16)
    """

    # 卷积权重转换: PyTorch [OC, IC, H, W] -> TensorRT [OC, IC, H, W] (保持OIHW格式，转换为fp16)
    if name.endswith('.weight') and len(tensor.shape) == 4:
        print(f"    转换卷积权重格式: {name} {list(tensor.shape)} (保持OIHW格式，转换为fp16)")
        return tensor.half()  # TensorRT使用标准OIHW格式

    # 全连接权重转换: 保持原始格式但转换为fp16 (ResNet的fc层)
    elif name.endswith('.weight') and len(tensor.shape) == 2 and 'fc' in name:
        print(f"    转换FC权重格式: {name} {list(tensor.shape)} (保持原始格式，转换为fp16)")
        return tensor.half()

    # BatchNorm权重转换为fp16
    elif name.endswith('.weight') and len(tensor.shape) == 1:
        print(f"    转换BN权重格式: {name} {list(tensor.shape)} (转换为fp16)")
        return tensor.half()

    # bias和其他参数转换为fp16
    elif name.endswith('.bias') or name.endswith('.running_mean') or name.endswith('.running_var'):
        print(f"    转换参数: {name} {list(tensor.shape)} (转换为fp16)")
        return tensor.half()

    # 其他权重转换为fp16
    return tensor.half()

def convert_pth_to_tensorrt_wts(pth_path, wts_path):
    """
    将PyTorch .pth文件转换为TensorRT兼容的 .wts 格式
    """
    print(f"Loading PyTorch weights from: {pth_path}")

    # 加载PyTorch权重文件
    try:
        checkpoint = torch.load(pth_path, map_location='cpu')

        # 如果是state_dict格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        print(f"Loaded {len(state_dict)} weight tensors")

    except Exception as e:
        print(f"Error loading PyTorch file: {e}")
        return False

    # 写入wts文件
    print(f"\nWriting TensorRT compatible weights to: {wts_path} (FP16)")

    try:
        with open(wts_path, 'w') as f:
            # 写入权重数量
            f.write(f"{len(state_dict)}\n")

            # 遍历每个权重张量
            for name, tensor in state_dict.items():
                print(f"  Converting: {name} ({tensor.shape})...")

                # 转换tensor为float32 (用于权重转换)
                tensor_float = tensor.float()

                # 应用权重转换优化 (包含fp16转换)
                tensor_converted = apply_weight_transforms_for_tensorrt(name, tensor_float)

                # 展平tensor
                tensor_flat = tensor_converted.flatten()

                # 写入名称和大小
                f.write(f"{name} {len(tensor_flat)}")

                # 批量处理避免OOM
                batch_size = 1000
                total_elements = len(tensor_flat)

                for i in range(0, total_elements, batch_size):
                    end_idx = min(i + batch_size, total_elements)
                    batch = tensor_flat[i:end_idx]

                    # 转换为FP16 hex
                    hex_values = []
                    for value in batch:
                        # FP16: 使用half精度打包
                        fp16_val = np.float16(value.item())
                        packed_bytes = fp16_val.tobytes()
                        hex_int = struct.unpack('H', packed_bytes)[0]
                        hex_values.append(f"{hex_int:x}")

                    # 写入文件
                    f.write(" " + " ".join(hex_values))

                f.write("\n")
                print(f"  ✓ Converted: {name}")

        print(f"\n✅ Successfully converted to: {wts_path}")
        return True

    except Exception as e:
        print(f"Error writing wts file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch ResNet50 .pth to TensorRT .wts format (FP16)')
    parser.add_argument('input', help='Input .pth file path')
    parser.add_argument('-o', '--output', help='Output .wts file path',
                       default='resnet50-tensorrt-fp16.wts')

    args = parser.parse_args()

    # 执行转换
    success = convert_pth_to_tensorrt_wts(args.input, args.output)

    if success:
        print(f"\n🎉 TensorRT conversion completed successfully!")
        print(f"You can now use: {args.output}")
    else:
        print(f"\n❌ Conversion failed!")

if __name__ == "__main__":
    main()