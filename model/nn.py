import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Identity):
    def __init__(self):
        super(Identity, self).__init__()
    
    def set_sample_config(self, *args, **kwargs):
        return None

    def get_params(self):
        params = 0
        return params
        

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        self.sample_in_features = in_features
        self.sample_out_features = out_features
    
    def set_sample_config(self, in_features, out_features):
        self.sample_in_features = in_features
        self.sample_out_features = out_features

    def sample_parameters(self):
        weight = self.weight[:self.sample_out_features, :self.sample_in_features]
        bias = self.bias[:self.sample_out_features] if self.bias is not None else None
        return weight, bias

    def get_params(self):
        weight, bias = self.sample_parameters()
        params = weight.numel()
        params += 0 if bias is None else bias.numel()
        return params

    def forward(self, x):
        weight, bias = self.sample_parameters()
        return F.linear(x, weight, bias)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.sample_in_channels = in_channels
        self.sample_out_channels = out_channels

    def set_sample_config(self, in_channels, out_channels):
        self.sample_in_channels = in_channels
        self.sample_out_channels = out_channels
    
    def sample_parameters(self):
        weight = self.weight[:self.sample_out_channels, :self.sample_in_channels]
        bias = self.bias[:self.sample_out_channels] if self.bias is not None else None
        return weight, bias

    def get_params(self):
        weight, bias = self.sample_parameters()
        params = weight.numel()
        params += 0 if bias is None else bias.numel()
        return params

    def forward(self, x):
        weight, bias = self.sample_parameters()
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.sample_num_features = num_features

    def set_sample_config(self, num_features):
        self.sample_num_features = num_features
    
    def sample_parameters(self):
        weight = self.weight[:self.sample_num_features] if self.weight is not None else None
        bias = self.bias[:self.sample_num_features] if self.bias is not None else None
        running_mean = self.running_mean[:self.sample_num_features] if self.running_mean is not None else None
        running_var = self.running_var[:self.sample_num_features] if self.running_var is not None else None
        return weight, bias, running_mean, running_var
    
    def get_params(self):
        weight, bias = self.sample_parameters()
        params = weight.numel()
        params += 0 if bias is None else bias.numel()
        return params

    def forward(self, x):
        weight, bias, running_mean, running_var = self.sample_parameters()
        return F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=self.training, momentum=self.momentum, eps=self.eps)


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(LayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.sampled_normalized_shape = self.normalized_shape
    
    def set_sample_config(self, normalized_shape):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )  
        self.sampled_normalized_shape = tuple([min(i, j) for i, j in zip(normalized_shape, self.normalized_shape)])

    def sample_parameters(self):
        indices = [slice(0, i) for i in self.sampled_normalized_shape]
        weight = self.weight[indices] if self.weight is not None else None
        bias = self.bias[indices] if self.bias is not None else None
        return weight, bias

    def get_params(self):
        weight, bias = self.sample_parameters()
        params = 0 if weight is None else weight.numel()
        params += 0 if bias is None else bias.numel()
        return params

    def forward(self, x):
        weight, bias = self.sample_parameters()
        return F.layer_norm(x, self.sampled_normalized_shape, weight, bias, eps=self.eps)






if __name__ == "__main__":
    model = Identity()
    x = torch.rand(4, 3, 8, 8)
    with torch.no_grad():
        y = model(x)
        print(x.shape, y.shape)
