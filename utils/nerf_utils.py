import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import get_encoder
from .activation import trunc_exp



class NeRFNetwork(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_light="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_vis=4,
                 hidden_dim_vis=4,               
                 bound=1,
                 **kwargs,
                 ):
        super().__init__()

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        self.bound = bound
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir,degree=4)

        self.encoder_light, self.in_dim_light = get_encoder(encoding_light,degree=4)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        
        color_net =  []
        # V3
        # for l in range(num_layers_color):
        #     if l == 0:
        #         # in_dim = self.in_dim_dir + self.geo_feat_dim
        #         # in_dim = self.in_dim_dir + self.geo_feat_dim + self.in_dim_light
        #         in_dim = self.in_dim_dir \
        #             + self.geo_feat_dim + self.in_dim_light + self.in_dim
        #     else:
        #         in_dim = hidden_dim_color
            
        #     if l == num_layers_color - 1:
        #         out_dim = 3 # 3 rgb
        #     else:
        #         out_dim = hidden_dim_color
            
        #     color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # V4
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir \
                    + self.in_dim_light + self.in_dim
            elif l==1:
                in_dim = hidden_dim_color + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        
        self.color_net = nn.ModuleList(color_net)

        # background network

        #V5
        self.num_layers_vis = num_layers_vis        
        self.hidden_dim_vis = hidden_dim_vis
        vis_net = []
        for l in range(num_layers_vis):
            if l == 0:
                in_dim = self.in_dim_light + self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1  # 1 binary masking
            else:
                out_dim = hidden_dim
            
            vis_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.vis_net = nn.ModuleList(vis_net)
        

    def forward(self, x, d, lig):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        pos = x
        # sigma
        x = self.encoder(x, bound=self.bound)
        h_x = x
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        # light = (lig-pos)/torch.linalg.norm(lig-pos)
        light = lig
        
        
        # light = l

        d = self.encoder_dir(d)
        light = self.encoder_light(light)
        # h = torch.cat([d, geo_feat], dim=-1)
        # h = torch.cat([light,d, geo_feat], dim=-1)
        
        
        # V3
        # h = torch.cat([h_x,light,d, geo_feat], dim=-1)
        # for l in range(self.num_layers_color):
        #     h = self.color_net[l](h)
        #     if l != self.num_layers_color - 1:
        #         h = F.relu(h, inplace=True)

        # V4
        # h = torch.cat([h_x,light,d], dim=-1)
        # for l in range(self.num_layers_color):
        #     if l == 1:
        #         h = torch.cat([h,geo_feat],dim=-1)
        #     h = self.color_net[l](h)
            
        #     if l != self.num_layers_color - 1:
        #         h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        # color = torch.sigmoid(h)
        # color = torch.relu(h)
        # color = self.leakyrelu(h)
        # color = self.tanh(h)

        # V5
        h = torch.cat([h_x,light,d], dim=-1)
        for l in range(self.num_layers_color):
            if l == 1:
                h = torch.cat([h,geo_feat],dim=-1)
            h = self.color_net[l](h)
            
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        color = torch.relu(h)
        
        h1 = torch.cat([h_x,light], dim=-1)
        for l in range(self.num_layers):
            h1 = self.vis_net[l](h1)
            if l != self.num_layers - 1:
                h1 = F.relu(h1, inplace=True)
        
        vis = torch.sigmoid(h1)

        col = color*vis
        # return sigma, color
        return col

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }


    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.vis_net.parameters(), 'lr': lr}
        ]
        
        
        return params


if __name__ == '__main__':
    model = NeRFNetwork(
        encoding="hashgrid",
        num_layers=2,
        num_layers_color=3,
        bound=2,
        cuda_ray=False,
        density_scale=1,
        min_near=0.2,
        density_thresh=10,
        bg_radius=-1,
    ).cuda()
    N = 60000
    x = torch.rand([N,3]).cuda()
    d = torch.rand([N,3]).cuda()
    l = torch.rand([N,3]).cuda()
    color = model(x,d,l)
    print(color)