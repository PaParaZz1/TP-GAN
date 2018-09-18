import sys
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

sys.path.append("..")
import config.constant as constant
import config.config as config
import network.TP_GAN as TP_GAN
import data_preprocessing.dataset as dataset

USE_CUDA = config.USE_CUDA

class GANTest():
	def __init__(self):
		
		self.img_size = constant.IMG_SIZE
		
		self.noise_dim = config.generator["noise_dim"]
		self.generator = TP_GAN.Generator(noise_dim = config.generator["noise_dim"],
			encode_feature_dim = config.generator["encode_feature_dim"], encode_predict_dim = config.generator["encode_predict_dim"],
			use_batchnorm = config.generator["use_batchnorm"], use_residual_block = config.generator["use_residual_block"])
		self.discriminator = TP_GAN.Discriminator(config.discriminator["use_batchnorm"])
		self.batch_size = config.test["batch_size"]
		self.test_dataloader = torch.utils.data.DataLoader(dataset.TestDataset(config.path["img_test"]), batch_size = config.test["batch_size"], shuffle = False, num_workers = 8)
		
		if USE_CUDA:
			self.generator = self.generator.cuda()
			self.discriminator = self.discriminator.cuda()
			self.generator.load_state_dict(torch.load(config.test["generator_model"]))
			self.discriminator.load_state_dict(torch.load(config.test["discriminator_model"]))
		else:
			self.generator.load_state_dict(torch.load(config.test["generator_model"], map_location = "cpu"))
			self.discriminator.load_state_dict(torch.load(config.test["discriminator_model"], map_location = "cpu"))
			
 
		def test(self):
			self.generator.eval()
			self.discriminator.eval()
			for batch_index, inputs in enumerate(self.train_dataloader):
				noise = Variable(torch.FloatTensor(np.random.uniform(-1, 1,(self.batch_size, self.noise_dim))))
				for k,v in inputs:
					if USE_CUDA:
						v = v.cuda()
					v = Variable(v, requires_grad = False)
				if USE_CUDA:
					noise = noise.cuda()
				generator_output = generator(inputs["img"], inputs["left_eye"], inputs["right_eye"], inputs["nose"], inputs["mouth"], noise)
				save_image(generator_output["img128_predict"])
		
		def save_image(self, img):
			raise Exception("method hasn't been implemented")
