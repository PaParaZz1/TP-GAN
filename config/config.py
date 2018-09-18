import torch

USE_CUDA = torch.cuda.is_available()

# generator
generator = {}
generator["noise_dim"] = 64
generator["encode_feature_dim"] = 256
generator["encode_predict_dim"] = 347
generator["use_batchnorm"] = False
generator["use_residual_block"] = False

# discriminator
discriminator = {}
discriminator["use_batchnorm"] = False

# settings
settings = {}
settings["init_lr"] = 1e-4
settings["epoch"] = 1e3
settings["batch_size"] = 1e2
 
# loss
loss = {}
loss["weight_gradient_penalty"] = 10
loss["pixelwise_weight_128"] = 1.0
loss["pixelwise_weight_64"] = 1.0
loss["pixelwise_weight_32"] = 1.5
loss["symmetry_weight_128"] = 1.0
loss["symmetry_weight_64"] = 1.0
loss["symmetry_weight_32"] = 1.5
loss["pixelwise_global_weight"] = 1.0
loss["pixelwise_local_weight"] = 3.0
loss["symmetry_weight"] = 3e-1
loss["adversarial_weight"] = 1e-3
loss["identity_preserving_weight"] = 3e1
loss["total_variation_weight"] = 1e-3
loss["cross_entropy_weight"] = 1e1

# path
path = {}
path["img_train"] = "dataset/img_train.list"
path["img_test"] = "dataset/img_test.list"
path["feature_extract_model_path"] = ""
path["generator_save_path"] = "model/generator"
path["discriminator_save_path"] = "model/discriminator"

# test(infernence)
test = {}
test["batch_size"] = 1
test["generator_model"] = "model/generator.pth"
test["discriminator_model"] = "model/discriminator.pth"
