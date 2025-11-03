#!/usr/bin/env python3
"""
ViT PyTorch to ONNX Converter
Converts PyTorch ViT model (.pth) to ONNX format for TensorRT trtexec
"""

import sys
import argparse
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from typing import Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ViT .pth file to .onnx')
    parser.add_argument('-w', '--weights', required=True,
                        help='Input weights (.pth) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.onnx) file path (optional)')
    parser.add_argument('-t', '--type', type=str, default='large', 
                        choices=['base', 'large'],
                        help='ViT model type: base or large (default: large)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for ONNX export (default: 1)')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--opset-version', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model after export')
    parser.add_argument('--verify', action='store_true',
                        help='Verify ONNX model after export')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.weights):
        raise SystemExit(f'ERROR: Input file not found: {args.weights}')
    
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.onnx'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.onnx')
    
    return args

def create_timm_vit():
    """创建timm兼容的ViT模型"""
    class PatchEmbed(nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.num_patches = (img_size // patch_size) ** 2
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
        def forward(self, x):
            B, C, H, W = x.shape
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x

    class Attention(nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
            
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

    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
            
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class ViT(nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                     embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                     qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
            super().__init__()
            self.num_classes = num_classes
            self.num_features = self.embed_dim = embed_dim
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
            num_patches = self.patch_embed.num_patches
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)
            self.blocks = nn.Sequential(*[Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate) for i in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            
        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            x = self.blocks(x)
            x = self.norm(x)
            x = self.head(x[:, 0])  # Extract CLS token
            return x

    return ViT

def load_model(pt_file: str, model_type: str) -> nn.Module:
    """加载PyTorch模型"""
    print(f"Loading ViT-{model_type.upper()} model from {pt_file}")
    
    # Create model based on type
    ViT = create_timm_vit()
    if model_type == 'large':
        model = ViT(img_size=224, patch_size=16, num_classes=1000, 
                   embed_dim=1024, depth=24, num_heads=16)
    else:  # base
        model = ViT(img_size=224, patch_size=16, num_classes=1000, 
                   embed_dim=768, depth=12, num_heads=12)
    
    # Load weights
    device = 'cpu'
    checkpoint = torch.load(pt_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    
    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def export_to_onnx(model: nn.Module, output_path: str, batch_size: int, 
                   input_size: int, opset_version: int) -> bool:
    """导出模型到ONNX格式"""
    print(f"Exporting model to ONNX format...")
    print(f"Output path: {output_path}")
    print(f"Batch size: {batch_size}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"ONNX opset version: {opset_version}")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        print(f"✅ ONNX export successful: {output_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

def verify_onnx_model(onnx_path: str) -> bool:
    """验证ONNX模型"""
    try:
        import onnx
        print(f"Verifying ONNX model: {onnx_path}")
        
        # Load and check model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print("✅ ONNX model verification passed")
        return True
    except ImportError:
        print("⚠️  onnx package not installed, skipping verification")
        return True
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")
        return False

def simplify_onnx_model(onnx_path: str) -> bool:
    """简化ONNX模型"""
    try:
        import onnxsim
        print(f"Simplifying ONNX model: {onnx_path}")
        
        # Load model
        import onnx
        model = onnx.load(onnx_path)
        
        # Simplify
        simplified_model, check = onnxsim.simplify(model)
        if check:
            onnx.save(simplified_model, onnx_path)
            print("✅ ONNX model simplification successful")
            return True
        else:
            print("❌ ONNX model simplification failed")
            return False
    except ImportError:
        print("⚠️  onnxsim package not installed, skipping simplification")
        return True
    except Exception as e:
        print(f"❌ ONNX model simplification failed: {e}")
        return False

def print_model_info(model: nn.Module, model_type: str):
    """打印模型信息"""
    print(f"\n=== ViT-{model_type.upper()} Model Information ===")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print model structure summary
    print(f"\nModel structure:")
    for name, module in model.named_children():
        print(f"  {name}: {module}")
    
    # Print input/output shapes
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("=" * 50)

def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"ViT PyTorch to ONNX Converter")
    print(f"Converting ViT-{args.type.upper()} model")
    print(f"{'='*60}\n")
    
    # Load model
    model = load_model(args.weights, args.type)
    
    # Print model information
    print_model_info(model, args.type)
    
    # Export to ONNX
    success = export_to_onnx(model, args.output, args.batch_size, 
                           args.input_size, args.opset_version)
    
    if not success:
        print("❌ Export failed!")
        return 1
    
    # Verify model if requested
    if args.verify:
        verify_onnx_model(args.output)
    
    # Simplify model if requested
    if args.simplify:
        simplify_onnx_model(args.output)
    
    # Print file size
    file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
    print(f"\n📁 ONNX file size: {file_size:.2f} MB")
    
    print(f"\n✅ Conversion completed successfully!")
    print(f"ONNX model saved to: {args.output}")
    
    # Print trtexec command
    print(f"\n{'='*60}")
    print(f"TensorRT trtexec Command:")
    print(f"{'='*60}")
    print(f"trtexec --onnx={args.output} \\")
    print(f"        --saveEngine=./vit-{args.type}-onnx.engine \\")
    print(f"        --fp16 \\")
    print(f"        --workspace=4096 \\")
    print(f"        --minShapes=images:1x3x{args.input_size}x{args.input_size} \\")
    print(f"        --optShapes=images:{args.batch_size}x3x{args.input_size}x{args.input_size} \\")
    print(f"        --maxShapes=images:{args.batch_size}x3x{args.input_size}x{args.input_size}")
    print(f"{'='*60}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

