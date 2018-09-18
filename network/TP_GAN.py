import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import OrderedDict
import network.modules as module

sys.path.append("..")
import config.constant as constant

''' hook tools for look up layer output

def hook(model, act_in, act_out):
	if torch.is_tensor(act_out):
		outputs = (act_out,)
	else:
		outputs = act_out
	for x in act_out:
		np.save("layer_name", x.data.cpu().numpy()	

self.layer_name.register_forward_hook(hook)
	
'''


class LocalPath(nn.Module):
	def __init__(self, feature_dim = 64, use_batchnorm = True):
		super(Local, self).__init__()
		channel_encoder = (64, 128, 256, 512)
		channel_decoder = (256, 128, feature_dim)
		# encoder
		leaky_value = 1e-2
		self.conv1 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = 3, out_channels = channel_encoder[0],
					kernel_size = 3, stride = 1, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(leaky_value), use_batchnorm = use_batchnorm),
					module.ResidualBlock(channel_encoder[0], activation = nn.LeakyReLU()))

		self.conv2 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_encoder[0], out_channels = channel_encoder[1],
					kernel_size = 3, stride = 2, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(leaky_value), use_batchnorm = use_batchnorm),
					module.ResidualBlock(channel_encoder[1], activation = nn.LeakyReLU()))

		self.conv3 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_encoder[1], out_channels = channel_encoder[2],
					kernel_size = 3, stride = 2, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(leaky_value), use_batchnorm = use_batchnorm),
					module.ResidualBlock(channel_encoder[2], activation = nn.LeakyReLU()))

		self.conv4 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_encoder[2], out_channels = channel_encoder[3],
					kernel_size = 3, stride = 2, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(leaky_value), use_batchnorm = use_batchnorm),
					module.ResidualBlock(channel_encoder[3], activation = nn.LeakyReLU()))

		# decoder
		self.deconv1 = module.DeConvBlockSequential(
					in_channels = channel_encoder[3], out_channels = channel_decoder[0],
					kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)
		
		self.decode1 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_decoder[0] + self.conv3.out_channels, out_channels = channel_decoder[0],
					kernel_size = 3, stride = 1, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(), use_batchnorm = use_batchnorm), 
					module.ResidualBlock(channel_decoder[0], activation = nn.LeakyReLU()))

		self.deconv2 = module.DeConvBlockSequential(
					in_channels = channel_decoder[0], out_channels = channel_decoder[1],
					kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)
		
		self.decode2 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_decoder[1] + self.conv2.out_channels, out_channels = channel_decoder[1],
					kernel_size = 3, stride = 1, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(), use_batchnorm = use_batchnorm), 
					module.ResidualBlock(channel_decoder[1], activation = nn.LeakyReLU()))

		self.deconv3 = module.DeConvBlockSequential(
					in_channels = channel_encoder[1], out_channels = channel_decoder[2],
					kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)
		
		self.decode3 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_decoder[2] + self.conv1.out_channels, out_channels = channel_decoder[2],
					kernel_size = 3, stride = 1, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(), use_batchnorm = use_batchnorm), 
					module.ResidualBlock(channel_decoder[2], activation = nn.LeakyReLU()))
		
		self.local_path_output = module.ConvBlockSequential(
							in_channels = channel_decoder[2], out_channels = 3,
							kernel_size = 1, stride = 1, padding = 0,
							init_type = None, activation = None, use_batchnorm = False)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)

		deconv1 = self.deconv1(conv4)
		decode1 = self.decode1(torch.cat([deconv1, conv3], 1))
		deconv2 = self.deconv2(decode1)
		decode2 = self.decode2(torch.cat([deconv2, conv2], 1))
		deconv3 = self.deconv3(decode2)
		decode3 = self.decode3(torch.cat([deconv3, conv1], 1))
		local_path_output = self.local_path_output(decode3)

		return local_path_output, deconv3

class LocalFuser(nn.Module):
	def __init__(self):
		super(LocalFuser, self).__init__()
	def forward(self, x):
		return 0	

class GlobalPath(nn.Module):
	def __init__(self, noise_dim, scaling_factor = 1., use_batchnorm = True, use_residual_block = True):
		super(GlobalPath, self).__init__()
		leaky_value = 1e-2
		self.input_image_size = constant.IMG_SIZE 
		self.use_residual_block = use_residual_block
		self.noise_dim = noise_dim
		fc_size = self.input_image_size / pow(2,4)
		channel_encoder = (64, 64, 128, 256, 512)
		channel_decoder_prepare = (fc_size*8, fc_size*4, fc_size*2, fc_size)
		channel_decoder = (512, 256, 128, 64)
		channel_decoder_conv = (64, 32)
		# encoder
		self.conv1 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = 3, out_channels = channel_encoder[0],
					kernel_size = 7, stride = 1, padding = 3, init_type = "kaiming",
					activation = nn.LeakyReLU(leaky_value), use_batchnorm = use_batchnorm),
					module.ResidualBlock(in_channels = channel_encoder[0], kernel_size = 7, padding = 3,
					activation = nn.LeakyReLU(leaky_value), scaling_factor = scaling_factor))
		
		self.conv2 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_encoder[0], out_channels = channel_encoder[1],
					kernel_size = 5, stride = 2, padding = 2, init_type = "kaiming",
					activation = nn.LeakyReLU(leaky_value), use_batchnorm = use_batchnorm),
					module.ResidualBlock(in_channels = channel_encoder[1], kernel_size = 5, padding = 2,
					activation = nn.LeakyReLU(leaky_value), scaling_factor = scaling_factor))
		
		self.conv3 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_encoder[1], out_channels = channel_encoder[2],
					kernel_size = 3, stride = 2, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(leaky_value), use_batchnorm = use_batchnorm),
					module.ResidualBlock(in_channels = channel_encoder[2], kernel_size = 3, padding = 1,
					activation = nn.LeakyReLU(leaky_value), scaling_factor = scaling_factor))
		
		self.conv4 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_encoder[2], out_channels = channel_encoder[3],
					kernel_size = 3, stride = 2, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(leaky_value), use_batchnorm = use_batchnorm),
					module.ResidualBlock(in_channels = channel_encoder[3], kernel_size = 3, padding = 1,
					activation = nn.LeakyReLU(leaky_value), scaling_factor = scaling_factor))

		self.conv5 = module.BlockSequential(module.ConvBlockSequential(
					in_channels = channel_encoder[3], out_channels = channel_encoder[4],
					kernel_size = 3, stride = 2, padding = 1, init_type = "kaiming",
					activation = nn.LeakyReLU(leaky_value), use_batchnorm = use_batchnorm),
					module.ResidualBlock(in_channels = channel_encoder[4], kernel_size = 3, padding = 1,
					activation = nn.LeakyReLU(leaky_value), scaling_factor = scaling_factor))
		
		fc_size = self.input_image_size / pow(2,4)
		self.fc1 = nn.Linear(channel_encoder[4]*fc_size*fc_size, channel_encoder[4])
		self.maxpool1 = nn.MaxPool1d(kernel_size = 2, stride = 2, padding = 0)

		# decoder
		self.prepare8 = module.DeConvBlockSequential(
					in_channels = channel_encoder[4]/2 + self.noise_dim, out_channels = channel_decoder_prepare[0],
					kernel_size = 8, stride = 1, padding = 0, output_padding = 0,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)	
		
		self.prepare32 = module.DeConvBlockSequential(
					in_channels = channel_decoder_prepare[0], out_channels = channel_decoder_prepare[1],
					kernel_size = 3, stride = 4, padding = 0, output_padding = 1,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)	

		self.prepare64 = module.DeConvBlockSequential(
					in_channels = channel_decoder_prepare[1], out_channels = channel_decoder_prepare[2],
					kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)	

		self.prepare128 = module.DeConvBlockSequential(
					in_channels = channel_decoder_prepare[2], out_channels = channel_decoder_prepare[3],
					kernel_size = 3, stride = 2, padding = 1, output_padding = 0,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)	
		decode_dim8 = self.prepare8.out_channels + self.conv5.out_channels
		self.decode8 = module.ResidualBlock(
					in_channels = decode_dim8, kernel_size = 2, stride = 1,
					padding = [1,0,1,0], activation = nn.LeakyReLU())
		self.reconstruct8 = module.BlockSequential(
					*[module.ResidualBlock(in_channels = decode_dim8,
					kernel_size = 2, stride = 1, padding = [1,0,1,0],
					activation = nn.LeakyReLU() ) for i in range(2)])

		self.deconv16 = module.DeConvBlockSequential(
					in_channels = self.reconstruct8.out_channels, out_channels = channel_decoder[0],
					kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)
		
		decode_dim16 = self.conv4.out_channels 
		self.decode16 = module.ResidualBlock(
					in_channels = decode_dim16, activation = nn.LeakyReLU())
		self.reconstruct16 = module.BlockSequential(
					*[module.ResidualBlock(in_channels = decode_dim16 + self.deconv16.out_channels,
					activation = nn.LeakyReLU() ) for i in range(2)])

		self.deconv32 = module.DeConvBlockSequential(
					in_channels = self.reconstruct16.out_channels, out_channels = channel_decoder[1],
					kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)
		
		decode_dim32 = self.prepare32.out_channels + self.conv3.out_channels + 3
		self.decode32 = module.ResidualBlock(
					in_channels = decode_dim32, activation = nn.LeakyReLU())
		self.reconstruct32 = module.BlockSequential(
					*[module.ResidualBlock(in_channels = decode_dim32 + self.deconv32.out_channels,
					activation = nn.LeakyReLU() ) for i in range(2)])

		self.decode_output32 = ConvBlockSequential(
					in_channels = self.reconstruct32.out_channels, out_channels = 3,
					kernel_size = 3, stride = 1, padding = 1,
					init_type = None, activation = None)

		self.deconv64 = module.DeConvBlockSequential(
					in_channels = self.reconstruct32.out_channels, out_channels = channel_decoder[2],	
					kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)

		decode_dim64 = self.prepare64.out_channels + self.conv2.out_channels + 3
		self.decode64 = module.ResidualBlock(
					in_channels = decode_dim64, kernel_size = 5, activation = nn.LeakyReLU())
		self.reconstruct64 = module.BlockSequential(
					*[module.ResidualBlock(in_channels = decode_dim64 + self.deconv64.out_channels + 3,
					activation = nn.LeakyReLU() ) for i in range(2)])
		
		self.decode_output64 = ConvBlockSequential(
					in_channels = self.reconstruct64.out_channels, out_channels = 3,
					kernel_size = 3, stride = 1, padding = 1,
					init_type = None, activation = None)
	
		self.deconv128 = module.DeConvBlockSequential(
					in_channels = self.reconstruct64.out_channels, out_channels = channel_decoder[3],	
					kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
					init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = use_batchnorm)

		decode_dim128 = self.prepare128.out_channels + self.conv1.out_channels + 3
		self.decode128 = module.ResidualBlock(
					in_channels = decode_dim128, kernel_size = 7, activation = nn.LeakyReLU())
		self.reconstruct128 = module.BlockSequential(
					module.ResidualBlock(in_channels = decode_dim128 + self.deconv128.out_channels + 3 + local_feature_dim + 3,
					kernel_size = 5, activation = nn.LeakyReLU()))

		self.decode_conv1 = module.BlockSequential(
					ConvBlockSequential(in_channels = self.reconstruct128.out_channels, 
					out_channels = channel_decoder_conv[0],	kernel_size = 5, stride = 1, padding = 2,
					init_type = "kaiming", activation = nn.LeakyReLU(), use_batchnorm = use_batchnorm),
					module.ResidualBlock(channel_decoder_conv[0], kernel_size = 3))
		self.decode_conv2 = ConvBlockSequential(
					in_channels = channel_decoder_conv[0], out_channels = channel_decoder_conv[1],
					kernel_size = 3, stride = 1, padding = 1, 
					init_type = "kaiming", activation = nn.LeakyReLU(), use_batchnorm = use_batchnorm)
		self.decode_output128 = ConvBlockSequential(
					in_channels = channel_decoder_conv[1], out_channels = 3,
					kernel_size = 3, stride = 1, padding = 1,
					init_type = None, activation = None)
	def forward(self, img, local_predict, local_feature, noise):
		img64 = cv2.resize(64,64)
		img32 = cv2.resize(32,32)
		#encode	
		conv1 = self.conv1(img)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)
		
		fc1 = self.fc1(conv5.view(conv5.size()[0], -1))
		maxpool1 = self.maxpool1(fc1.view(fc1.size()[0], -1, 2).view(fc1.size()[0], -1))
		#decode
		prepare8 = self.prepare8(torch.cat[maxpool1, noise], 1).view(maxpool1.size()[0], -1, 1, 1)
		prepare32 = self.prepare32(prepare8)
		prepare64 = self.prepare64(prepare32)
		prepare128 = self.prepare128(prepare64)
		
		decode8 = self.decode8(torch.cat([prepare8, conv5], 1))
		reconstruct8 = self.reconstruct8(decode8)

		deconv16 = self.deconv16(reconstruct8)
		decode16 = self.decode16(conv4)
		reconstruct16 = self.reconstruct16(torch.cat([decode16, deconv16], 1))

		deconv32 = self.deconv32(reconstruct16)
		decode32 = self.decode32(torch.cat([prepare32, conv3, img32], 1))
		reconstruct32 = self.reconstruct32(torch.cat([decode32, deconv32], 1))
		output32 = self.decode_output32(reconstruct32)

		deconv64 = self.deconv64(reconstruct32)
		decode64 = self.decode64(torch.cat([prepare64, conv2, img64], 1))
		upsample64 = F.upsample(output32.data, (64,64), mode = "bilinear")
		reconstruct64 = self.reconstruct64(torch.cat([decode64, deconv64, upsample64], 1))
		output64 = self.decode_output64(reconstruct64)

		deconv128 = self.deconv128(reconstruct64)
		decode128 = self.decode128(torch.cat([prepare128, conv1, img], 1))
		upsample128 = F.upsample(output64.data, (128,128), mode = "bilinear")
		reconstruct128 = self.reconstruct128(torch.cat([decode128, deconv128, upsample128, local_feature, local_predict], 1))
		
		decode_conv1 = self.decode_conv1(reconstruct128)
		decode_conv2 = self.decode_conv2(decode_conv1)
		output128 = self.decode_output128(decode_conv2)

		return output128, output64, output32, maxpool1

class EncodePredict(nn.Module):
	def __init__(self, encode_feature_dim, output_dim, dropout_ratio = 0.3):
		super(EncodePredict, self).__init__()
		self.encode_feature_dim = encode_feature_dim
		self.dropout = nn.Dropout(p = dropout_ratio)	
		self.fc = nn.Linear(encode_feature_dim, output_dim)

	def forward(self, encode_feature, use_dropout):
		if use_dropout:
			encode_feature = self.dropout(encode_feature)
		encode_predict = self.fc(encode_feature)
		return encode_predict
		
class Generator(nn.Module):
	def __init__(self, noise_dim, encode_feature_dim, encode_predict_dim, use_batchnorm = True, use_residual_block = True):
		super(Generator, self).__init__()
		# local path
		self.local_path_left_eye = LocalPath(use_batchnorm = use_batchnorm)
		self.local_path_right_eye = LocalPath(use_batchnorm = use_batchnorm)
		self.local_path_nose = LocalPath(use_batchnorm = use_batchnorm)
		self.local_path_mouth = LocalPath(use_batchnorm = use_batchnorm)

		# global path and fuse
		self.global_path = GlobalPath(noise_dim, use_batchnorm = use_batchnorm, use_residual_block = use_residual_block)
		self.local_fuser = LocalFuser()
		self.encode_predict = EncodePredict(encode_feature_dim, encode_predict_dim)

	def forward(self, img, left_eye, right_eye, nose, mouth, noise, use_dropout = True):
		# local path
		left_eye_predict, left_eye_feature = self.local_path_left_eye(left_eye)
		right_eye_predict, right_eye_feature = self.local_path_right_eye(right_eye)
		nose_predict, nose_feature = self.local_path_nose(nose)
		mouth_predict, mouth_feature = self.local_path_mouth(mouth)

		# fuse
		local_feature = self.local_fuser(left_eye_feature, right_eye_feature, nose_feature, mouth_feature)
		local_predict = self.local_fuser(left_eye_predict, right_eye_predict, nose_predict, mouth_predict)
		local_input = self.local_fuser(left_eye, right_eye, nose, mouth)

		# global path
		img128_predict, img64_predict, img32_predict, encode_feature = self.global_path(img, local_predict, local_feature, noise)
		encode_predict = self.encode_predict(encode_feature, use_dropout)
		result = {"img128_predict":img128_predict, "img64_predict":img64_predict, "img32_predict":img32_predict, "encode_predict":encode_predict, "local_predict":local_predict,
			"left_eye_predict":left_eye_predict, "right_eye_predict":right_eye_predict, "nose_predict":nose_predict, "mouth_predict":mouth_predict, "local_input":local_input}
		return result

class Discriminator(nn.Module):
	def __init__(self, use_batchnorm = False):
		super(Discriminator, self).__init__()
		layers = OrderedDict()
		channel_output = (3, 64, 128, 256, 512, 512)
		leaky_value = 1e-2
		for i in range(5):
			layers["conv_block{}".format(i+1)] = module.ConvBlockSequential(
				in_channels = channel_output[i], out_channels = channel_output[i+1],
				kernel_size = 3, stride = 2, padding = 1,
				init_type = "kaiming", activation = nn.LeakyReLU(), use_batchnorm = use_batchnorm)
			if i>=3:
				layers["residual_block{}".format(i+1)] = module.ResidualBlock(
					in_channels = channel_output[i+1], activation = nn.LeakyReLU())
		layers["conv_last"] = module.ConvBlockSequential(
			in_channels = channel_output[-1], out_channels = 1, kernel_size = 3, stride = 1,
			padding = 1, init_type = None, activation = None)
		self.model = module.BlockSequential(**layers)

	def forward(self, x):
		return self.model(x)
