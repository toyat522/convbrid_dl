import math

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F

class ConvAttentionMask:
    def __init__(self,
                 num_token: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 include_cls_token: bool = True):
        
        self.num_token = num_token

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.include_cls_token = include_cls_token

        self.bundled_tokens_indices = self._get_bundled_indices()
        self.attention_mask = torch.full((self.num_token, self.num_token), float('-inf'))
        for bundled_indices in self.bundled_tokens_indices:
            self._add_bundle_into_attention_mask(bundled_indices)

        if include_cls_token:
            self.bundled_tokens_indices = [bundled_indices + (bundled_indices >= 0).to(torch.int32) for bundled_indices in self.bundled_tokens_indices]
            self.attention_mask = F.pad(self.attention_mask, (1, 0, 1, 0), value=1)
        else:
            self.attention_mask = self.attention_mask.fill_diagonal_(1)

        self.attention_mask = F.pad(self.attention_mask, (0, 1, 0, 1), value=0)

    def _get_bundled_indices(self):
        self.image_size = int(math.sqrt(self.num_token))
        referal_indices_tensor = torch.arange(self.num_token).reshape(self.image_size, self.image_size)
        referal_indices_tensor = F.pad(referal_indices_tensor, (0,self.padding,0,self.padding), value=-1)

        central_tokens_1d_coords = torch.arange(self.kernel_size // 2 - self.padding, self.image_size - self.kernel_size // 2 + self.padding, self.stride)
        central_tokens_2d_coords = torch.cartesian_prod(central_tokens_1d_coords, central_tokens_1d_coords).reshape(-1, 1, 2)

        bundled_tokens_2d_coords = self._get_bundled_coords(central_tokens_2d_coords)

        return [referal_indices_tensor[coords[:, 0], coords[:, 1]] for coords in bundled_tokens_2d_coords]

    def _get_bundled_coords(self,central_coords):
        kernel_extend = torch.arange(- (self.kernel_size // 2), self.kernel_size // 2 + 1)
        kernel_coords = torch.cartesian_prod(kernel_extend, kernel_extend)

        return central_coords + kernel_coords.reshape(1, -1, 2)
    
    def _add_bundle_into_attention_mask(self,bundled_indices):
        bundled_indices = bundled_indices[bundled_indices >= 0]
        attention_coords = torch.cartesian_prod(bundled_indices, bundled_indices)
        self.attention_mask[attention_coords[:, 0], attention_coords[:, 1]] = 1


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = num_heads ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,x,attn_mask=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_scores = attn_scores * attn_mask
            print(attn_scores)
            
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        x = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class MLP(nn.Module):
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
    
class ConvAttention(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, padding=0,
                 num_heads=1, include_cls_token=True, mlp_ratio=4.,
                 qkv_bias=None, attn_drop=0., proj_drop=0.,
                 drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.include_cls_token = include_cls_token

        self.norm1 = norm_layer(in_channels)
        self.attn = Attention(in_channels, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = norm_layer(in_channels)
        self.pooling = nn.Linear(int(kernel_size**2 * in_channels),in_channels)

        self.norm3 = norm_layer(in_channels)
        self.mlp = MLP(in_features=in_channels, 
                       hidden_features=int(in_channels * mlp_ratio),
                       out_features=out_channels,
                       act_layer=act_layer, drop=drop)
        
    def forward(self, x, conv_mode=False):
        B, N, C = x.shape

        x = self.norm1(x)

        if conv_mode:
            N_tokens = N-1 if self.include_cls_token else N

            conv_utils = ConvAttentionMask(N_tokens, self.kernel_size, self.stride, self.padding, self.include_cls_token)
            attn_mask = conv_utils.attention_mask.to(x.device)

            x = F.pad(x, (0,0,0,1,0,0), mode='constant', value=0)
            x = x + self.attn(x, attn_mask)

            x = x[:, :-1, :]
            x = self.norm2(x)

            x_tokens = torch.cat([x[:,bundled_indices,:] for bundled_indices in conv_utils.bundled_tokens_indices], dim=1).reshape(B, len(conv_utils.bundled_tokens_indices), -1)
            x_tokens = self.pooling(x_tokens)

            if self.include_cls_token:
                x_cls = x[:,0,:].unsqueeze(1)
                x = torch.cat([x_cls, x_tokens], dim=1)
            else:
                x = x_tokens

        else:
            x = x + self.attn(x)

        x = self.mlp(self.norm3(x))

        return x
    

class ConvTradition(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 include_cls_token=False,mlp_ratio=4.,
                 drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.act = act_layer()

        self.norm_conv1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2)

        self.norm_mlp1 = norm_layer(in_channels)
        self.mlp1 = MLP(in_features=in_channels,
                        hidden_features=int(in_channels * mlp_ratio),
                        out_features=in_channels,
                        act_layer=act_layer, drop=drop)
        
        self.norm_conv2 = norm_layer(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        
        self.norm_mlp2 = norm_layer(in_channels)
        self.mlp2 = MLP(in_features=in_channels,
                        hidden_features=int(in_channels * mlp_ratio),
                        out_features=out_channels,
                        act_layer=act_layer, drop=drop)
        
    def _tokens_to_pixels(self,x):
        B,N,C = x.shape
        return x.reshape(B, int(math.sqrt(N)), int(math.sqrt(N)), C).permute(0,3,1,2)
    
    def _pixels_to_tokens(self,x):
        B,C,D,D = x.shape
        return x.permute(0,2,3,1).reshape(B,D**2,C)

    def forward(self, x, conv_mode=True):
        x = self.norm_conv1(x)
        x = self._tokens_to_pixels(x)
        x = self.conv1(x)
        x = self.act(x)

        x = x + self.mlp1(self.norm_mlp1(x,self._pixels_to_tokens(x)))

        x = self.norm_conv2(x)
        x = self._tokens_to_pixels(x)
        x = self.conv2(x)
        x = self.act(x)

        x = self.mlp2(self.norm_mlp2(x,self._pixels_to_tokens(x)))

        return x


if __name__ == '__main__':
    x = torch.randn(10, 16, 16)
    
    test_trans = ConvAttention(in_channels=16,out_channels=32,kernel_size=5,stride=2,padding=2,num_heads=1,include_cls_token=False)
    test_conv = ConvTradition(in_channels=16,out_channels=32,kernel_size=5,stride=2,padding=2,include_cls_token=False)
    print(test_trans.forward(x))
    #print(test_conv.forward(x))