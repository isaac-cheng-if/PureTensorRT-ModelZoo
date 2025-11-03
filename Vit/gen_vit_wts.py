#!/usr/bin/env python3
"""
ViT Weight Converter
Converts PyTorch ViT model weights to .wts format for TensorRT
"""

import sys
import argparse
import os
import struct
import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ViT .pth file to .wts')
    parser.add_argument('-w', '--weights', required=True,
                        help='Input weights (.pth) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
    parser.add_argument('-t', '--type', type=str, default='base', 
                        choices=['base', 'large'],
                        help='ViT model type: base or large')
    args = parser.parse_args()
    
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    
    return args.weights, args.output, args.type

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
        def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
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
            x = self.head(x[:, 0])
            return x

    return ViT

def main():
    pt_file, wts_file, model_type = parse_args()
    
    print(f'Generating .wts for ViT-{model_type.upper()} model')
    print(f'Loading {pt_file}')
    
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
    
    print(f'Model loaded successfully')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Convert to .wts format
    print(f'Converting to .wts format...')
    with open(wts_file, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            # 保持原始张量形状，按行优先顺序展平
            vr = v.flatten().cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
    
    print(f'✅ .wts file generated: {wts_file}')
    print(f'Total tensors: {len(model.state_dict().keys())}')

if __name__ == '__main__':
    main()
