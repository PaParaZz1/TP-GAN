import torch.nn as nn
from torch.nn.init import xavier_normal, kaiming_normal

def WeightInitialize(init_type, weight, activation = None):
	if init_type is None:
		return
	def XavierInit(weight):
		xavier_normal(weight)
	def KaimingInit(weight, activation):
		assert not activation is None
		if hasattr(activation, "negative_slope"):
			kaiming_normal(weight, a = activation.negative_slope)
		else:
			kaiming_normal(weight, a = 0)

	init_type_dict = {"xavier" : XavierInit, "kaiming" : KaimingInit}
	if init_type in init_type_dict:
		init_type_dict[init_type](weight, activation)
	else:
		raise KeyError("Invalid Key:%s" % init_type)

def ConvBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = False):
	block_conv = []
	# (TODO) deal with special(2D) padding
	block_conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
	WeightInitialize(init_type, block_conv[-1].weight, activation)
	if not activation is None:
		block_conv.append(activation)
	if use_batchnorm:
		block_conv.append(nn.BatchNorm2d(out_channels))
	return block_conv

def ConvBlockSequential(in_channels, out_channels, kernel_size, stride = 1, padding = 0, init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = False):
	seq = nn.Sequential(*ConvBlock(in_channels, out_channels, kernel_size, stride, padding, init_type, activation, use_batchnorm))
	seq.out_channels = out_channels
	return req

def DeConvBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, output_padding = 0, init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = False):
	block_deconv = []
	block_deconv.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding))
	WeightInitialize(init_type, block_deconv[-1].weight, activation)
	if not activation is None:
		block_deconv.append(activation)
	if use_batchnorm:
		block_deconv.append(nn.BatchNorm2d(out_channels))
	return block_deconv
	
def DeConvBlockSequential(in_channels, out_channels, kernel_size, stride = 1, padding = 0, output_padding = 0, init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = False):
	seq =  nn.Sequential(*DeConvBlock(in_channels, out_channels, kernel_size, stride, padding, output_padding, init_type, activation, use_batchnorm))
	seq.out_channels = out_channels
	return req

def BlockSequential(*kargs):
	seq = nn.Sequential(*kargs)
	for layer in reversed(kargs):
		if hasattr(layer, "out_channels"):
			seq.out_channels = layer.out_channels
			break
		if hasattr(layer, "out_features"):
			seq.out_channels = layer.out_features
			break
	return seq

class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels = None, kernel_size = 3, stride = 1, \
				padding = None, init_type = "kaiming", activation = nn.ReLU(), \
				is_bottleneck = False, use_projection = False, scaling_factor= 1.):
		super(type(self), self).__init__()
		if out_channels is None:
			out_channels = in_channels // stride
	
		self.activation = activation
		self.scaling_factor = scaling_factor
		assert stride in [1,2]
		if stride == 1:
			self,shortcut = nn.Sequential()
		else:
			self.shortcut = ConvBlockSequential(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0, init_type = None, activation = None, use_batchnorm = False)
		
		# (TODO) use projection
	
		block = []
		if is_bottleneck:
			block.append(ConvBlockSequential(in_channels = in_channels, out_channels = in_channels // 2, kernel_size = 1, stride = 1, padding = 0, init_type = init_type, activation = activation, use_batchnorm = False))
			block.append(ConvBlockSequential(in_channels // 2, out_channels // 2, kernel_size, stride, padding = (kernel_size - 1)// 2, init_type = init_type, activation = activation, use_batchnorm = False))
			block.append(ConvBlockSequential(out_channels // 2, out_channels, kernel_size = 1, stride = 1, padding = 0, init_type = None, activation = None, use_batchnorm = False))
		else:
			if padding is None:
				padding = (kernel_size - 1)// 2
			block.append(ConvBlockSequential(in_channels, in_channels, kernel_size, stride = 1, padding = padding, init_type = init_type, activation = activation, use_batchnorm = False))
			block.append(ConvBlockSequential(in_channels, out_channels, kernel_size, stride = 1, padding = padding, init_type = None, activation = None, use_batchnorm = False))
		self.block = nn.Sequential(*block)

	def forward(self, x):
		return self.activation(self.block(x) + self.scaling_factor * self.shortcut(x))
		
