import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================
# AFBS + K/V Frequency Enhancement + SFDF
# ============================================
class AFBSTSR(nn.Module):
    def __init__(self, up_scale=4, dim=60, groups=5, num=4):
        super(AFBSTSR, self).__init__()

        self.init = nn.Conv2d(
            in_channels=3,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True
        )

        self.num = num
        self.groups = groups
        self.body = nn.ModuleList()

        for i in range(groups):
            self.body.append(
                GroupSR(
                    dim=dim,
                    num=num,
                    wsize=16
                )
            )

        self.up = nn.Sequential(
            nn.Conv2d(
                in_channels=dim,
                out_channels=3 * up_scale ** 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True
            ),
            nn.PixelShuffle(up_scale)
        )

        self.up_scale = up_scale

    def forward(self, x0):
        x = self.init(x0)

        for i in range(self.groups):
            x = self.body[i](x)

        x = self.up(x) + F.interpolate(
            x0,
            scale_factor=self.up_scale,
            mode='bilinear',
            align_corners=False
        )

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data

                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('up') == -1:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'
                            .format(name, own_state[name].size(), param.size())
                        )
            elif strict:
                if name.find('up') == -1:
                    raise KeyError(
                        'unexpected key "{}" in state_dict'.format(name)
                    )

# ============================================
# Residual Group
# ============================================
class GroupSR(nn.Module):
    def __init__(self, dim, num=6, wsize=16):
        super().__init__()

        self.num = num
        self.body = nn.ModuleList()

        for i in range(num):
            self.body.append(
                BasicBlockSR(
                    dim=dim,
                    spatial=(i % 2 == 0),
                    channel=(i % 2 == 1),
                    window_sizes=wsize,
                    shift=((i + 2) % 4 == 0)
                )
            )

        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x0):
        x = x0

        for i in range(self.num):
            x = self.body[i](x)

        x = self.conv(x)

        return x + x0


# ============================================
# Basic Block
# ============================================
class BasicBlockSR(nn.Module):
    def __init__(
        self,
        dim,
        spatial=False,
        channel=False,
        window_sizes=8,
        shift=False
    ):
        super().__init__()

        self.spatial = SelfAttention(
            dim=dim,
            heads=2,
            wsize=window_sizes,
            shift=shift
        ) if spatial else None

        self.channel = DWBlcok(
            dim=dim,
            wsize=8
        ) if channel else None

        self.window_sizes = window_sizes
        self.MLP = MLP(dim, ratio=2)

    def check_image_size2(self, x, wsize):
        _, _, h, w = x.size()

        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize

        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        if self.spatial:
            x = self.check_image_size2(x, self.window_sizes)
            x = self.spatial(x)[:, :, :h, :w]

        if self.channel:
            x = self.check_image_size2(x, 8)
            x = self.channel(x)[:, :, :h, :w]

        x = self.MLP(x)

        return x
# ============================================
# FCEM：Frequency Channel Enhancement Mmodule
# ============================================
class DWBlcok(nn.Module):
    def __init__(self, dim, wsize=8):
        super().__init__()

        self.norm = LayerNorm2d(dim)

        self.projin = nn.Conv2d(dim, dim * 2, 1)
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.q = nn.Conv2d(dim, dim, 1)
        self.wsize = wsize
        self.proj = nn.Conv2d(dim, dim, 1)

        self.filter = nn.Parameter(
            torch.randn(dim, wsize, wsize // 2 + 1, 2) * 0.02
        )

    def forward(self, x0):
        b, c, h, w = x0.shape

        x = self.norm(x0)

        x1, x2 = self.projin(x).chunk(2, dim=1)

        x1 = rearrange(
            x1,
            'b c (h dh) (w dw) -> (b h w) c dh dw',
            dh=self.wsize,
            dw=self.wsize
        )

        x1 = torch.fft.rfft2(x1, norm='ortho')

        q_r = self.q(x1.real)
        q_i = self.q(x1.imag)

        x1 = torch.complex(q_r, q_i) * torch.view_as_complex(self.filter)

        x1 = torch.fft.irfft2(x1, s=(self.wsize, self.wsize), norm='ortho')

        x1 = rearrange(
            x1,
            '(b h w) c dh dw -> b c (h dh) (w dw)',
            b=b,
            h=h // self.wsize,
            w=w // self.wsize
        )

        x = x1 * self.local(x2)
        x = self.proj(x)

        return x + x0


# ============================================
# AFBS: Adaptive Frequency Band Selection
# ============================================
class AdaptiveFrequencyBandSelection(nn.Module):
    """
    # 输入:
    #    x: [B, C, H, W]

    # 过程:
    #     1. 局部窗口 FFT
    #     2. 低频 / 中频 / 高频三个频带分别滤波
    #     3. 根据输入窗口内容动态生成 band gate
    #    4. 自适应融合三个频带
    #     5. IFFT 回到空间域

    # 输出:
    #     x_out: [B, C, H, W]
    """
    def __init__(self, dim, patch_size=4, hidden_ratio=4):
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size

        hidden = max(dim // hidden_ratio, 8)

        self.pre = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        # 三个频带的可学习复数滤波器
        self.low_filter = nn.Parameter(
            torch.randn(dim, patch_size, patch_size // 2 + 1, 2) * 0.02
        )
        self.mid_filter = nn.Parameter(
            torch.randn(dim, patch_size, patch_size // 2 + 1, 2) * 0.02
        )
        self.high_filter = nn.Parameter(
            torch.randn(dim, patch_size, patch_size // 2 + 1, 2) * 0.02
        )

        # # 根据每个局部窗口的内容生成低/中/高频 gate
        self.band_gate = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 1, 1, 0, bias=True)
        )

        self.norm = LayerNorm2d(dim)
        self.post = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        low_mask, mid_mask, high_mask = self._build_frequency_masks(patch_size)

        self.register_buffer("low_mask", low_mask, persistent=False)
        self.register_buffer("mid_mask", mid_mask, persistent=False)
        self.register_buffer("high_mask", high_mask, persistent=False)

        self.last_band_gate = None
        self.last_fft_mag = None
        self.last_low_response = None
        self.last_mid_response = None
        self.last_high_response = None
        self.last_freq_feat = None

    def _build_frequency_masks(self, patch_size):
        #"""
      #  为 rfft2 输出构建低 / 中 / 高频 mask。
        #        # rfft2 输出形状为 [patch_size, patch_size//2 + 1]
      #  """
        fy = torch.fft.fftfreq(patch_size)
        fx = torch.fft.rfftfreq(patch_size)

        yy, xx = torch.meshgrid(fy, fx, indexing='ij')

        radius = torch.sqrt(xx ** 2 + yy ** 2)
        radius = radius / (radius.max() + 1e-8)

        low = (radius <= 0.34).float()
        mid = ((radius > 0.34) & (radius <= 0.67)).float()
        high = (radius > 0.67).float()

        low = low.view(1, 1, patch_size, patch_size // 2 + 1)
        mid = mid.view(1, 1, patch_size, patch_size // 2 + 1)
        high = high.view(1, 1, patch_size, patch_size // 2 + 1)

        return low, mid, high

    def forward(self, x):
        b, c, h, w = x.shape
        ps = self.patch_size

        x = self.pre(x)

        # [B, C, H, W] -> [B, C, H//ps, W//ps, ps, ps]
        x_patch_6d = rearrange(
            x,
            'b c (nh ph) (nw pw) -> b c nh nw ph pw',
            ph=ps,
            pw=ps
        )

        #  band gate
        # [B, C, nh, nw]
        gate_feat = x_patch_6d.mean(dim=(-1, -2))

        # [B, 3, nh, nw]
        band_gate = self.band_gate(gate_feat)
        band_gate = torch.softmax(band_gate, dim=1)

        self.last_band_gate = band_gate.detach()

        # [B, C, nh, nw, ps, ps] -> [B*nh*nw, C, ps, ps]
        x_patch = rearrange(
            x_patch_6d,
            'b c nh nw ph pw -> (b nh nw) c ph pw'
        )

        x_fft = torch.fft.rfft2(x_patch.float(), norm='ortho')

        self.last_fft_mag = torch.abs(x_fft).detach()

        low_filter = torch.view_as_complex(self.low_filter)
        mid_filter = torch.view_as_complex(self.mid_filter)
        high_filter = torch.view_as_complex(self.high_filter)

        # [1, C, ps, ps//2+1]
        low_filter = low_filter.unsqueeze(0)
        mid_filter = mid_filter.unsqueeze(0)
        high_filter = high_filter.unsqueeze(0)

        low_response = x_fft * low_filter * self.low_mask
        mid_response = x_fft * mid_filter * self.mid_mask
        high_response = x_fft * high_filter * self.high_mask

        self.last_low_response = torch.abs(low_response).detach()
        self.last_mid_response = torch.abs(mid_response).detach()
        self.last_high_response = torch.abs(high_response).detach()

        # band_gate: [B, 3, nh, nw]
        # -> [(B nh nw), 3, 1, 1, 1]
        gate = rearrange(
            band_gate,
            'b three nh nw -> (b nh nw) three 1 1 1',
            three=3
        )

        x_fft = (
            gate[:, 0] * low_response +
            gate[:, 1] * mid_response +
            gate[:, 2] * high_response
        )

        x_ifft = torch.fft.irfft2(x_fft, s=(ps, ps), norm='ortho')

        x_ifft = rearrange(
            x_ifft,
            '(b nh nw) c ph pw -> b c (nh ph) (nw pw)',
            b=b,
            nh=h // ps,
            nw=w // ps,
            ph=ps,
            pw=ps
        )

        x_ifft = self.norm(x_ifft)
        x_ifft = self.post(x_ifft)

        self.last_freq_feat = x_ifft.detach()

        return x_ifft

# ============================================
# SFDF: Spatial-Frequency Dynamic Fusion
# ============================================
class SpatialFrequencyDynamicFusion(nn.Module):
   """
   INPUT:
       xs: attention branch, [B, C, H, W]
       xf: frequency branch, [B, C, H, W]

   OUTPUT:
       out: [B, C, H, W]

   gate :
       1. spatial gate
   #    2. channel gate
   """
   def __init__(self, dim, hidden_ratio=4):
       super().__init__()

       hidden = max(dim // hidden_ratio, 8)

       self.spatial_gate = nn.Sequential(
           nn.Conv2d(dim * 2, hidden, 1, 1, 0, bias=True),
           nn.GELU(),
           nn.Conv2d(hidden, 1, 3, 1, 1, bias=True),
           nn.Sigmoid()
       )

       self.channel_gate = nn.Sequential(
           nn.AdaptiveAvgPool2d(1),
           nn.Conv2d(dim * 2, hidden, 1, 1, 0, bias=True),
           nn.GELU(),
           nn.Conv2d(hidden, dim, 1, 1, 0, bias=True),
           nn.Sigmoid()
        )
       self.out_proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
       self.last_gate = None
       self.last_spatial_gate = None
       self.last_channel_gate = None

   def forward(self, xs, xf):
       fusion_input = torch.cat([xs, xf], dim=1)

       spatial = self.spatial_gate(fusion_input)
       channel = self.channel_gate(fusion_input)

       gate = 0.5 * (spatial + channel)

       self.last_spatial_gate = spatial.detach()
       self.last_channel_gate = channel.detach()
       self.last_gate = gate.detach()

       out = gate * xs + (1.0 - gate) * xf
       out = self.out_proj(out)

       return out

# ============================================
# AFAM:Adaptive Frequency Attention Module
# ============================================
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=1, wsize=8, shift=False):
        super().__init__()

        adim = dim // 2

        self.norm = LayerNorm2d(dim)
        self.adim = adim

        self.qkv = nn.Conv2d(dim, 3 * adim, 1)

        self.wsize = wsize
        self.patch_size = wsize // 4

        self.scale = (adim // heads) ** -0.5
        self.shift = shift
        self.softmax = nn.Softmax(dim=-1)
        self.heads = heads

        self.proj = nn.Conv2d(adim, dim, 1)
        self.filter = nn.Parameter(torch.randn(adim, wsize//4, wsize // 8 + 1, 2)*0.02)
        # K
        self.k_afbs = AdaptiveFrequencyBandSelection(
            dim=adim,
            patch_size=self.patch_size
        )

        self.local = nn.Sequential(
           nn.Conv2d(dim, adim, 1),
           nn.Conv2d(adim, adim, 3, 1, 1, groups=adim)
        )

        # SFDF
        self.sfdf = SpatialFrequencyDynamicFusion(adim)

        self.gamma_k = nn.Parameter(torch.zeros(1))


    def forward(self, x0):
        b, c, h, w = x0.shape

        x = self.norm(x0)

        qkv = self.qkv(x)

        q0, k0, v0 = qkv.chunk(3, dim=1)

        # =====================================================
        # K AFBS
        # =====================================================
        k_freq = self.k_afbs(k0)

        self.last_k_freq = k_freq.detach()
        self.last_gamma_k = self.gamma_k.detach()

        k0 = k0 + self.gamma_k * k_freq

        qkv = torch.cat([q0, k0, v0], dim=1)

        v1 = qkv[:,-self.adim:,:,:]
        local = rearrange(v1,'b c (h dh) (w dw)->(b h w) c dh dw',dh=self.wsize//4,dw =self.wsize//4)
        local = torch.fft.rfft2(local,norm='ortho')
        weight = torch.view_as_complex(self.filter)
        local = local * weight
        local = torch.fft.irfft2(local,norm='ortho')
        local = rearrange(local,'(b h w) c dh dw->b c (h dh) (w dw)',b=b,h=h//self.wsize*4,w = w//self.wsize*4)
        local = local * self.local(x)

        # =====================================================
        # 3. shifted window self-attention
        # =====================================================
        if self.shift:
            qkv_attn = torch.roll(
                qkv,
                shifts=(-self.wsize // 2, -self.wsize // 2),
                dims=(2, 3)
            )
        else:
            qkv_attn = qkv

        q, k, v = qkv_attn.chunk(3, dim=1)

        q = rearrange(
            q,
            'b (hed c) (h dh) (w dw) -> (b h w) hed (dh dw) c',
            dh=self.wsize,
            dw=self.wsize,
            hed=self.heads
        )

        k = rearrange(
            k,
            'b (hed c) (h dh) (w dw) -> (b h w) hed (dh dw) c',
            dh=self.wsize,
            dw=self.wsize,
            hed=self.heads
        )

        v = rearrange(
            v,
            'b (hed c) (h dh) (w dw) -> (b h w) hed (dh dw) c',
            dh=self.wsize,
            dw=self.wsize,
            hed=self.heads
        )

        atn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        atn = self.softmax(atn)
        y = torch.matmul(atn, v)

        y = rearrange(
            y,
            '(b h w) hed (dh dw) c -> b (hed c) (h dh) (w dw)',
            h=h // self.wsize,
            w=w // self.wsize,
            dh=self.wsize,
            dw=self.wsize
        )

        if self.shift:
            y = torch.roll(
                y,
                shifts=(self.wsize // 2, self.wsize // 2),
                dims=(2, 3)
            )

        # =====================================================
        # SFDF
        # =====================================================
        y = self.sfdf(y, local)
        y = self.proj(y)

        return y + x0


# ============================================
# MLP
# ============================================
class MLP(nn.Module):
    def __init__(self, dim, ratio=2):
        super(MLP, self).__init__()

        self.layernorm1 = LayerNorm2d(dim)

        expandim = int(dim * ratio)

        self.proj1 = nn.Conv2d(dim, expandim, 1)
        self.conv = nn.Conv2d(expandim, expandim, 3, 1, 1, groups=expandim)
        self.projout = nn.Conv2d(dim, dim, 1)

    def forward(self, x0):
        x = self.layernorm1(x0)

        x = self.proj1(x)

        x1, x2 = self.conv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2

        x = self.projout(x)

        return x + x0


# ============================================
# LayerNorm2d
# ============================================
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super(LayerNorm2d, self).__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h, w = x.shape[-2:]

        x = to_3d(x)

        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

        return to_4d(x, h, w)

if __name__ == "__main__":
    net = AFBSTSR(up_scale=4, dim=60, groups=5, num=4)

    total = sum([param.nelement() for param in net.parameters()])
    print('Number of params: %.2fK' % (total / 1e3))

    x = torch.randn(1, 3, 64, 64)
    y = net(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)