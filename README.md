# Lightweight Transformer Network for Inertial Navigation
Code for paper: "LTNIN:Lightweight Transformer Network for Inertial Navigation"

## Prerequisites

Install dependency use pip:
```bash
pip install fvcore
pip install timm
pip install thop
```

## Usage
The LTNIN is contained in LightNiT .py, and the core code is as follows, by setting the number of layers of blocks to 1, 2, and 4 corresponding to LightNiT-S, LightNiT-R, and LightNiT, respectively
````python
class LightNiT(nn.Module):
    def __init__(self, seq_len=200, in_chans=6, num_classes=2, embed_dim=128, num_heads=8, mlp_ratio=5., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = ConvStem1D(seq_len=seq_len, in_chans=in_chans, embed_dim=embed_dim)

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, act_layer=act_layer
            )
            for _ in range(4)  # number of HybridAttnBlock   LightNiT:4  LightNiT-S:1 LightNiT-R:2
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x, _ = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

````

Example output:
```bash
x: torch.Size([128,2])
```

## Acknowledgements
Thanks for RONIN [https://github.com/Sachini/ronin].

## License
The source code is released under GPLv3 license.