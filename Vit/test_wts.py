#!/usr/bin/env python3
"""
ViT WTS文件验证脚本
验证转换后的.wts文件是否可以正确进行图片推理
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import struct

def load_wts_weights(wts_file):
    """加载.wts文件中的权重"""
    print(f"📥 加载.wts文件: {wts_file}")
    
    weight_map = {}
    
    with open(wts_file, 'r') as f:
        # 读取张量数量
        count = int(f.readline().strip())
        print(f"📊 总张量数量: {count}")
        
        for i in range(count):
            line = f.readline().strip()
            parts = line.split(' ')
            
            if len(parts) < 3:
                continue
                
            name = parts[0]
            size = int(parts[1])
            
            # 读取权重数据
            weights = []
            for j in range(2, len(parts)):
                if parts[j]:  # 跳过空字符串
                    # 将十六进制字符串转换为float
                    hex_str = parts[j]
                    if len(hex_str) == 8:  # 确保是4字节的十六进制
                        float_val = struct.unpack('>f', bytes.fromhex(hex_str))[0]
                        weights.append(float_val)
            
            if len(weights) == size:
                weight_map[name] = torch.tensor(weights, dtype=torch.float32)
                print(f"✅ 加载张量: {name}, 形状: {len(weights)}")
            else:
                print(f"❌ 张量大小不匹配: {name}, 期望: {size}, 实际: {len(weights)}")
    
    print(f"📦 成功加载 {len(weight_map)} 个张量")
    return weight_map

def create_timm_vit():
    """创建timm兼容的ViT模型"""
    class PatchEmbed(torch.nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.num_patches = (img_size // patch_size) ** 2
            self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
        def forward(self, x):
            B, C, H, W = x.shape
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x

    class Attention(torch.nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = head_dim ** -0.5
            self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = torch.nn.Dropout(attn_drop)
            self.proj = torch.nn.Linear(dim, dim)
            self.proj_drop = torch.nn.Dropout(proj_drop)
            
        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class Mlp(torch.nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=torch.nn.GELU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = torch.nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = torch.nn.Linear(hidden_features, out_features)
            self.drop = torch.nn.Dropout(drop)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class Block(torch.nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.norm2 = torch.nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=torch.nn.GELU, drop=drop)
            
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class ViT(torch.nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
            super().__init__()
            self.num_classes = num_classes
            self.num_features = self.embed_dim = embed_dim
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
            num_patches = self.patch_embed.num_patches
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = torch.nn.Dropout(p=drop_rate)
            self.blocks = torch.nn.Sequential(*[Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate) for i in range(depth)])
            self.norm = torch.nn.LayerNorm(embed_dim)
            self.head = torch.nn.Linear(self.num_features, num_classes) if num_classes > 0 else torch.nn.Identity()
            
        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            x = self.blocks(x)
            x = self.norm(x)
            x = self.head(x[:, 0])
            return x

    return ViT

def load_wts_to_model(model, weight_map, model_type):
    """将.wts权重加载到模型中"""
    print(f"🔄 加载权重到ViT-{model_type.upper()}模型...")
    
    loaded_count = 0
    total_params = 0
    
    # 使用 named_parameters() 获取实际参数的引用，而不是副本
    for name, param in model.named_parameters():
        total_params += 1
        if name in weight_map:
            # 获取权重张量
            wts_tensor = weight_map[name]
            
            # 重塑张量形状以匹配模型参数
            if wts_tensor.numel() == param.numel():
                param.data = wts_tensor.view_as(param)
                loaded_count += 1
                print(f"✅ 加载: {name}, 形状: {param.shape}")
            else:
                print(f"❌ 形状不匹配: {name}, 模型: {param.shape}, WTS: {wts_tensor.shape}")
        else:
            print(f"⚠️  未找到权重: {name}")
    
    # 同时加载 buffers (如 LayerNorm 的 running stats)
    for name, buffer in model.named_buffers():
        if name in weight_map:
            wts_tensor = weight_map[name]
            if wts_tensor.numel() == buffer.numel():
                buffer.data = wts_tensor.view_as(buffer)
                loaded_count += 1
                print(f"✅ 加载buffer: {name}, 形状: {buffer.shape}")
    
    print(f"📊 成功加载 {loaded_count} 个参数")
    return loaded_count > 0

def compare_with_original():
    """与原始PyTorch模型比较"""
    print("🔍 与原始PyTorch模型比较...")
    
    # 加载原始模型
    ViT = create_timm_vit()
    original_model = ViT(img_size=224, patch_size=16, num_classes=1000, 
                        embed_dim=768, depth=12, num_heads=12)
    
    # 加载原始权重
    checkpoint = torch.load('../jx_vit_base_p16_224-80ecf9dd.pth', map_location='cpu', weights_only=True)
    original_model.load_state_dict(checkpoint)
    original_model.eval()
    
    # 加载WTS权重
    weight_map = load_wts_weights('vit_base.wts')
    wts_model = ViT(img_size=224, patch_size=16, num_classes=1000, 
                   embed_dim=768, depth=12, num_heads=12)
    load_wts_to_model(wts_model, weight_map, "base")
    wts_model.eval()
    
    # 测试图片
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open('../kitten.jpg').convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # 比较输出
    with torch.no_grad():
        original_output = original_model(input_tensor)
        wts_output = wts_model(input_tensor)
        
        print(f"📊 原始模型输出形状: {original_output.shape}")
        print(f"📊 WTS模型输出形状: {wts_output.shape}")
        
        # 计算差异
        diff = torch.abs(original_output - wts_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"🔍 最大差异: {max_diff:.6f}")
        print(f"🔍 平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            print("✅ WTS权重与原始权重完全匹配!")
        elif max_diff < 1e-3:
            print("✅ WTS权重与原始权重基本匹配!")
        else:
            print("❌ WTS权重与原始权重存在较大差异!")
            
        # 显示Top5预测
        original_probs = torch.softmax(original_output, dim=1)
        wts_probs = torch.softmax(wts_output, dim=1)
        
        original_top5 = torch.topk(original_probs, 5)
        wts_top5 = torch.topk(wts_probs, 5)
        
        print("\n🎯 原始模型Top5预测:")
        for i, (prob, idx) in enumerate(zip(original_top5[0][0], original_top5[1][0])):
            print(f"   {i+1}. 类别 {idx.item():3d}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
            
        print("\n🎯 WTS模型Top5预测:")
        for i, (prob, idx) in enumerate(zip(wts_top5[0][0], wts_top5[1][0])):
            print(f"   {i+1}. 类别 {idx.item():3d}: {prob.item():.4f} ({prob.item()*100:.2f}%)")

def test_inference(model, image_path):
    """测试模型推理"""
    print(f"🖼️  测试图片: {image_path}")
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误: 图片文件不存在: {image_path}")
        return False
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载和预处理图片
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    print(f"📐 输入形状: {input_tensor.shape}")
    
    # 推理
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    print(f"📊 输出形状: {output.shape}")
    print(f"🎯 Top 5 预测结果:")
    for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_indices[0])):
        print(f"   {i+1}. 类别 {idx.item():3d}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    
    return True

def main():
    print("=" * 60)
    print("ViT WTS文件验证测试")
    print("=" * 60)
    
    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python3 test_wts.py <wts_file> [model_type] [image_path]")
        print("示例: python3 test_wts.py vit_base.wts base kitten.jpg")
        print("或者: python3 test_wts.py compare  # 与原始模型比较")
        return
    
    if sys.argv[1] == "compare":
        # 与原始模型比较
        compare_with_original()
        return
    
    wts_file = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "base"
    image_path = sys.argv[3] if len(sys.argv) > 3 else "kitten.jpg"
    
    print(f"📁 WTS文件: {wts_file}")
    print(f"🏗️  模型类型: ViT-{model_type.upper()}")
    print(f"🖼️  测试图片: {image_path}")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(wts_file):
        print(f"❌ 错误: WTS文件不存在: {wts_file}")
        return
    
    # 加载.wts权重
    weight_map = load_wts_weights(wts_file)
    print()
    
    # 创建模型
    print("🏗️  创建ViT模型...")
    ViT = create_timm_vit()
    
    if model_type.lower() == "large":
        model = ViT(img_size=224, patch_size=16, num_classes=1000, 
                   embed_dim=1024, depth=24, num_heads=16)
    else:  # base
        model = ViT(img_size=224, patch_size=16, num_classes=1000, 
                   embed_dim=768, depth=12, num_heads=12)
    
    print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 加载权重到模型
    success = load_wts_to_model(model, weight_map, model_type)
    print()
    
    if not success:
        print("❌ 权重加载失败!")
        return
    
    # 测试推理
    print("🚀 开始推理测试...")
    test_inference(model, image_path)
    
    print()
    print("=" * 60)
    print("✅ WTS文件验证完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
