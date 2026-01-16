"""
BoxWorld ViT 双流网络稳定性预测模型
Dual-Stream Vision Transformer with Cross-Attention Fusion

架构:
- Global Stream (ViT): 全局视图 + Mask通道注入，理解堆叠结构
- Local Stream (ViT): 局部裁剪，观察接触点细节
- Cross-Attention Fusion: Local CLS 作为 Query，Global Patches 作为 Key/Value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PatchEmbed(nn.Module):
    """图像分块嵌入层，支持4通道输入(RGB+Mask)"""

    def __init__(self, img_size=224, patch_size=16, in_chans=4, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """多头自注意力"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """交叉注意力模块

    Local CLS token 作为 Query
    Global Patch tokens 作为 Key/Value
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        """
        Args:
            query: (B, 1, C) - Local CLS token
            key_value: (B, N, C) - Global patch tokens
        Returns:
            (B, 1, C) - 融合后的特征
        """
        B, _, C = query.shape
        N = key_value.shape[1]

        q = self.q_proj(query).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """前馈网络"""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerStream(nn.Module):
    """单个ViT流

    Args:
        img_size: 输入图像大小
        patch_size: patch大小
        in_chans: 输入通道数 (4 = RGB + Mask)
        embed_dim: 嵌入维度
        depth: Transformer块数量
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层比例
        return_all_tokens: 是否返回所有patch tokens
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=4, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., return_all_tokens=False):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.embed_dim = embed_dim

        # Patch嵌入
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias,
                             drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]

        # Patch嵌入
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer编码
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.return_all_tokens:
            # 返回所有tokens (用于Cross-Attention)
            return x  # (B, num_patches+1, embed_dim)
        else:
            # 只返回CLS token
            return x[:, 0]  # (B, embed_dim)


class CrossAttentionFusion(nn.Module):
    """Cross-Attention融合模块

    物理含义：Local CLS token 询问 Global patches
    "我在局部看到了这个纹理，请问这个边缘对应在全局图里的哪个位置？
     那个位置周围的力学结构稳不稳定？"
    """

    def __init__(self, embed_dim=768, num_heads=8, num_layers=2, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': CrossAttention(embed_dim, num_heads,
                                             qkv_bias=True, attn_drop=drop, proj_drop=drop),
                'norm1': nn.LayerNorm(embed_dim),
                'mlp': MLP(embed_dim, int(embed_dim * 4), drop=drop),
                'norm2': nn.LayerNorm(embed_dim)
            }))

    def forward(self, local_cls, global_tokens):
        """
        Args:
            local_cls: (B, embed_dim) - Local stream的CLS token
            global_tokens: (B, N, embed_dim) - Global stream的所有patch tokens
        Returns:
            (B, embed_dim) - 融合后的特征
        """
        # 扩展local_cls为 (B, 1, embed_dim)
        x = local_cls.unsqueeze(1)

        for layer in self.layers:
            # Cross-Attention
            x = x + layer['cross_attn'](layer['norm1'](x), global_tokens)
            # FFN
            x = x + layer['mlp'](layer['norm2'](x))

        return x.squeeze(1)  # (B, embed_dim)


class ViTTwoStreamStabilityPredictor(nn.Module):
    """ViT双流稳定性预测模型

    架构:
    - Global Stream: 输入全图+Mask，返回所有patch tokens
    - Local Stream: 输入局部裁剪，返回CLS token
    - Cross-Attention Fusion: Local CLS查询Global patches
    - MLP Head: 输出稳定性概率

    输入:
        global_input: (B, 4, 224, 224) - 全局视图 RGB + Mask
        local_input: (B, 3, 224, 224) - 局部裁剪 RGB (无需Mask)

    输出: (B, 1) - 稳定概率
    """

    def __init__(self, model_size='small', pretrained=True, drop_rate=0.1):
        """
        Args:
            model_size: 'tiny', 'small', 'base'
            pretrained: 是否加载预训练权重(仅对Local Stream有效)
            drop_rate: Dropout率
        """
        super().__init__()

        # 模型配置
        configs = {
            'tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
            'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
            'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        }
        cfg = configs.get(model_size, configs['small'])
        self.embed_dim = cfg['embed_dim']

        # Global Stream: 4通道输入(RGB+Mask)，返回所有tokens
        self.global_stream = VisionTransformerStream(
            img_size=224, patch_size=16, in_chans=4,
            embed_dim=cfg['embed_dim'], depth=cfg['depth'],
            num_heads=cfg['num_heads'], mlp_ratio=4.,
            drop_rate=drop_rate, attn_drop_rate=drop_rate,
            return_all_tokens=True
        )

        # Local Stream: 3通道输入(RGB)，只返回CLS token
        self.local_stream = VisionTransformerStream(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=cfg['embed_dim'], depth=cfg['depth'],
            num_heads=cfg['num_heads'], mlp_ratio=4.,
            drop_rate=drop_rate, attn_drop_rate=drop_rate,
            return_all_tokens=False
        )

        # Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            num_layers=2,
            drop=drop_rate
        )

        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(cfg['embed_dim']),
            nn.Linear(cfg['embed_dim'], 256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 加载预训练权重
        if pretrained:
            self._load_pretrained(model_size)

    def _load_pretrained(self, model_size):
        """加载timm预训练权重到Local Stream"""
        try:
            import timm
            model_names = {
                'tiny': 'vit_tiny_patch16_224',
                'small': 'vit_small_patch16_224',
                'base': 'vit_base_patch16_224',
            }
            name = model_names.get(model_size, 'vit_small_patch16_224')
            pretrained_vit = timm.create_model(name, pretrained=True)

            # 复制权重到Local Stream (3通道)
            self._copy_weights(pretrained_vit, self.local_stream)

            # 复制权重到Global Stream (需要扩展第一层)
            self._copy_weights_4ch(pretrained_vit, self.global_stream)

            print(f"Loaded pretrained weights from {name}")
        except ImportError:
            print("timm not installed, using random initialization")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    def _copy_weights(self, src_model, dst_stream):
        """复制预训练权重到目标stream (3通道)"""
        # Patch embedding
        dst_stream.patch_embed.proj.weight.data.copy_(
            src_model.patch_embed.proj.weight.data
        )
        dst_stream.patch_embed.proj.bias.data.copy_(
            src_model.patch_embed.proj.bias.data
        )

        # CLS token 和位置编码
        dst_stream.cls_token.data.copy_(src_model.cls_token.data)
        dst_stream.pos_embed.data.copy_(src_model.pos_embed.data)

        # Transformer blocks
        for i, blk in enumerate(dst_stream.blocks):
            src_blk = src_model.blocks[i]
            # Attention
            blk.attn.qkv.weight.data.copy_(src_blk.attn.qkv.weight.data)
            blk.attn.qkv.bias.data.copy_(src_blk.attn.qkv.bias.data)
            blk.attn.proj.weight.data.copy_(src_blk.attn.proj.weight.data)
            blk.attn.proj.bias.data.copy_(src_blk.attn.proj.bias.data)
            # MLP
            blk.mlp.fc1.weight.data.copy_(src_blk.mlp.fc1.weight.data)
            blk.mlp.fc1.bias.data.copy_(src_blk.mlp.fc1.bias.data)
            blk.mlp.fc2.weight.data.copy_(src_blk.mlp.fc2.weight.data)
            blk.mlp.fc2.bias.data.copy_(src_blk.mlp.fc2.bias.data)
            # LayerNorm
            blk.norm1.weight.data.copy_(src_blk.norm1.weight.data)
            blk.norm1.bias.data.copy_(src_blk.norm1.bias.data)
            blk.norm2.weight.data.copy_(src_blk.norm2.weight.data)
            blk.norm2.bias.data.copy_(src_blk.norm2.bias.data)

        # Final norm
        dst_stream.norm.weight.data.copy_(src_model.norm.weight.data)
        dst_stream.norm.bias.data.copy_(src_model.norm.bias.data)

    def _copy_weights_4ch(self, src_model, dst_stream):
        """复制预训练权重到4通道stream，第4通道初始化为0"""
        # Patch embedding: 扩展到4通道
        with torch.no_grad():
            dst_stream.patch_embed.proj.weight[:, :3, :, :].copy_(
                src_model.patch_embed.proj.weight.data
            )
            dst_stream.patch_embed.proj.weight[:, 3:, :, :].zero_()
        dst_stream.patch_embed.proj.bias.data.copy_(
            src_model.patch_embed.proj.bias.data
        )

        # CLS token 和位置编码
        dst_stream.cls_token.data.copy_(src_model.cls_token.data)
        dst_stream.pos_embed.data.copy_(src_model.pos_embed.data)

        # Transformer blocks (与3通道相同)
        for i, blk in enumerate(dst_stream.blocks):
            src_blk = src_model.blocks[i]
            blk.attn.qkv.weight.data.copy_(src_blk.attn.qkv.weight.data)
            blk.attn.qkv.bias.data.copy_(src_blk.attn.qkv.bias.data)
            blk.attn.proj.weight.data.copy_(src_blk.attn.proj.weight.data)
            blk.attn.proj.bias.data.copy_(src_blk.attn.proj.bias.data)
            blk.mlp.fc1.weight.data.copy_(src_blk.mlp.fc1.weight.data)
            blk.mlp.fc1.bias.data.copy_(src_blk.mlp.fc1.bias.data)
            blk.mlp.fc2.weight.data.copy_(src_blk.mlp.fc2.weight.data)
            blk.mlp.fc2.bias.data.copy_(src_blk.mlp.fc2.bias.data)
            blk.norm1.weight.data.copy_(src_blk.norm1.weight.data)
            blk.norm1.bias.data.copy_(src_blk.norm1.bias.data)
            blk.norm2.weight.data.copy_(src_blk.norm2.weight.data)
            blk.norm2.bias.data.copy_(src_blk.norm2.bias.data)

        dst_stream.norm.weight.data.copy_(src_model.norm.weight.data)
        dst_stream.norm.bias.data.copy_(src_model.norm.bias.data)

    def forward(self, global_input, local_input):
        """
        Args:
            global_input: (B, 4, 224, 224) - 全局视图 RGB + Mask
            local_input: (B, 3, 224, 224) - 局部裁剪 RGB
        Returns:
            (B, 1) - 稳定性概率
        """
        # Global Stream: 获取所有patch tokens
        global_tokens = self.global_stream(global_input)  # (B, N+1, embed_dim)

        # Local Stream: 获取CLS token
        local_cls = self.local_stream(local_input)  # (B, embed_dim)

        # Cross-Attention Fusion
        fused = self.fusion(local_cls, global_tokens)  # (B, embed_dim)

        # 分类
        out = self.head(fused)  # (B, 1)

        return out


if __name__ == "__main__":
    # 测试模型
    print("Testing ViT Two-Stream Model...")

    model = ViTTwoStreamStabilityPredictor(model_size='small', pretrained=False)

    # 模拟输入
    global_input = torch.randn(2, 4, 224, 224)  # RGB + Mask
    local_input = torch.randn(2, 3, 224, 224)   # RGB only

    output = model(global_input, local_input)
    print(f"Output shape: {output.shape}")  # (2, 1)
    print(f"Output: {output}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
