#README

This is a development pytorch version of TP-GAN(Two pathway GAN).

# requires
 - python3.6
 - pytorch0.3.1

# directory structure
 - config: global configuration and parameter
 - main: network train and test
 - utils: train support logs and functions
 - network: (TP-GAN)network architecture
 - data_preprocessing: data preprocessing 
 - data: image and image list
 - model: pre-train model, save model, feature_extract_model

# network architecture change
 - use WGAN-GP loss replace origin adversarial loss
 - use pytorch official resnet18 as feature extract network to calculate identity preserving loss

If you have any problems, please contact with niuyazhe@buaa.edu.cn
