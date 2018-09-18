import sys
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

sys.path.append("..")
import config.constant as constant
import config.config as config
import network.TP_GAN as TP_GAN
import network.resnet as resnet
import data_preprocessing.dataset as dataset

USE_CUDA = config.USE_CUDA


def set_requires_grad(model, flag):
	for parm in model.parameters():
		parm.requires_grad = flag


class GANTrain():
	def __init__(self):
		
		self.img_size = constant.IMG_SIZE
		
		self.noise_dim = config.generator["noise_dim"]
		self.generator = nn.DataParallel(TP_GAN.Generator(noise_dim = config.generator["noise_dim"],
			encode_feature_dim = config.generator["encode_feature_dim"], encode_predict_dim = config.generator["encode_predict_dim"],
			use_batchnorm = config.generator["use_batchnorm"], use_residual_block = config.generator["use_residual_block"]))
		self.discriminator = nn.DataParallel(TP_GAN.Discriminator(config.discriminator["use_batchnorm"]))
		self.lr = config.settings["init_lr"]
		self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr = self.lr)
		self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr = self.lr)
		self.lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_G, milestones = [], gamma = 0.1) 
		self.lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_D, milestones = [], gamma = 0.1) 
		self.epoch = config.settings["epoch"]
		self.batch_size = config.settings["batch_size"]
		self.train_dataloader = torch.utils.data.DataLoader(dataset.train_dataloader(config.path["img_test"], batch_size = self.batch_size, shuffle = True, num_workers = 8, pin_memory = True ))
		self.feature_extract_network_resnet18 = nn.DataParallel(resnet.resnet18(pretrained = True, num_classes = config.generator["encode_predict_dim"]))
		self.l1_loss = nn.L1Loss()
		self.MSE_loss = nn.MSELoss()
		self.cross_entropy = nn.CrossEntropyLoss()
		
		if USE_CUDA:
			self.generator = self.generator.cuda()
			self.discriminator = self.discriminator.cuda()
			self.l1_loss = self.l1_loss.cuda()
			self.MSE_loss = self.MSE_loss.cuda()
			self.cross_entropy = self.cross_entropy.cuda()
		
		def backward_D(self, img_predict, img_frontal):
			set_requires_grad(self.discriminator, True)
			adversarial_D_loss = -torch.mean(discriminator(img_frontal)) + torch.mean(discriminator(img_predict))
			factor = torch.rand(img_frontal.shape[0], 1, 1, 1).expand(img_frontal.size())#
			interpolated_value = Variable(factor * img_predict.data + (1.0-factor) * img_frontal.data, requires_grad = True)
			output = self.discriminator(interpolated_value)
			# WGAN-GP loss	
			gradient = torch.autograd.grad(outputs = output, inputs = interpolated_value, 
				grad_outputs = torch.ones(output.size()), retain_graph = True, create_graph = True, only_inputs = True)[0]
			gradient = gradient.view(output.shape[0], -1)
			if USE_CUDA:
				gradient = gradient.cuda()
			gp_loss = torch.mean((torch.norm(gradient, p=2) - 1) ** 2) 
			loss_D = adversarial_D_loss + config.loss["weight_gradient_penalty"] * gp_loss
			
			self.optimizer_D.zero_grad()
			loss_D.backward()
			self.optimizer_D.step()
			
			return loss_D

		def backward_G(self, inputs, outputs):
			set_requires_grad(self.discriminator, False)
			# adversarial loss
			adversarial_G_loss = -torch.mean(self.discriminator(outputs["img128_predict"]))
			# pixel wise loss
			pixelwise_loss_128 = self.l1_loss(inputs["img128_frontal"], outputs["img128_predict"])
			pixelwise_loss_64 = self.l1_loss(inputs["img64_frontal"], outputs["img64_predict"])
			pixelwise_loss_32 = self.l1_loss(inputs["img32_frontal"], outputs["img32_predict"])
			pixelwise_global_loss = config.loss["pixelwise_weight_128"] * pixelwise_weight_128 + \
				config.loss["pixelwise_weight_64"] * pixelwise_loss_64 + config.loss["pixelwise_weight_32"] * pixelwise_loss_32;

			left_eye_loss = self.l1_loss(inputs["left_eye_frontal"], outputs["left_eye_predict"])
			right_eye_loss = self.l1_loss(inputs["right_eye_frontal"], outputs["right_eye_predict"])
			nose_loss = self.l1_loss(inputs["nose_frontal"], outputs["nose_predict"])
			mouth_loss = self.l1_loss(inputs["mouth_frontal"], outputs["mouth_predict"])
			pixel_local_loss = left_eye_loss + right_eye_loss + nose_loss + mouth_loss
			# symmetry loss
			img128 = outputs["img128_predict"]
			img64 = outputs["img64_predict"]
			img32 = outputs["img32_predict"]
			inv_idx128 = torch.arange(img128.size()[3]-1, -1, -1).long()
			inv_idx64 = torch.arange(img64.size()[3]-1, -1,-1).long()
			inv_idx32 = torch.arange(img32.size()[3]-1, -1,-1).long()
			if USE_CUDA:
				inv_idx128 = inv_idx128.cuda()
				inv_idx64 = inv_idx64.cuda()
				inv_idx32 = inv_idx32.cuda()
			img128_flip = img128.index_select(3, Variable(inv_idx128))
			img64_flip = img64.index_select(3, Variable(inv_idx64))
			img32_flip = img32.index_select(3, Variable(inv_idx32))
			img128_flip.detach_()
			img64_flip.detach_()
			img32_flip.detach_()

			symmetry_loss_128 = self.l1_loss(img128, img128_flip)
			symmetry_loss_64 = self.l1_loss(img64, img64_flip)
			symmetry_loss_32 = self.l1_loss(img32, img32_flip)
			symmetry_loss = config.loss["symmetry_weight_128"] * symmetry_weight_128 + \
				config.loss["symmetry_weight_64"] * symmetry_weight_64 + config.loss["symmetry_weight_32"] * symmetry_weight_32
			
			# identity preserving loss
			feature_frontal = self.feature_extract_network_resnet18(inputs["img128_frontal"])
			feature_predict = self.feature_extract_network_resnet18(outputs["img128_predict"])
			identity_preserving_loss = self.MSE_loss(feature_frontal, feature_predict)

			# total variation loss for regularization
			img128 = outputs["img128_predict"]
			total_variation_loss = torch.mean(torch.abs(img128[:,:,:-1,:] - img128[:,:,1:,:])) + torch.mean(torch.abs(img128[:,:,:,:-1], img128[:,:,:,1:]))
			# cross entropy loss
			cross_entropy_loss = self.cross_entropy_loss(outputs["encode_predict"], inputs["label"])

			# synthesized loss
			synthesized_loss = config.loss["pixelwise_global_weight"] * pixelwise_global_loss + config.loss["pixel_local_weight"] * pixel_local_loss + \
				config.loss["symmetry_weight"] * symmetry_loss + config.loss["adversarial_G_weight"] * adversarial_G_loss + \
			 	config.loss["identity_preserving_weight"] * identity_preserving_loss + config.loss["total_variation_weight"] * total_variation_loss
			loss_G = synthesized_loss + config.loss["cross_entropy_weight"] * cross_entropy_loss
			
			self.optimizer_G.zero_grad()
			loss_G.backward()
			optimizer_G.step()

 
		def train(self):
			self.generator.train()
			self.discriminator.train()
			self.feature_extract_network_resnet18.eval()
			for cur_epoch in range(self.epoch):
				print("\ncurrent epoch number: %d"%cur_epoch)
				self.lr_scheduler_G.step()
				self.lr = self.lr_scheduler_G.get_lr()[0]
				self.lr_scheduler_D.step()
				self.lr = self.lr_scheduler_D.get_lr()[0]
				for batch_index, inputs in enumerate(self.train_dataloader):
					noise = Variable(torch.FloatTensor(np.random.uniform(-1, 1,(self.batch_size, self.noise_dim))))
					for k,v in inputs:
						if USE_CUDA:
							v = v.cuda()
						v = Variable(v, requires_grad = False)
					if USE_CUDA:
						noise = noise.cuda()
					generator_output = generator(inputs["img"], inputs["left_eye"], inputs["right_eye"], inputs["nose"], inputs["mouth"], noise)
					# backward
					backward_D(generator_output["img128_predict"].detach(), inputs["img128_frontal"])
					backward_G(generator_output, inputs)
			save_model()
	
		def save_model(self):
			torch.save(self.generator.cpu().state_dict(), config.path["generator__path"])
			torch.save(self.discriminator.cpu().state_dict(), config.path["discriminator_save_path"])	
