from torch.utils.data import Dataset


class TrainDataset(Dataset):
	def __init__(self, img_train_list):
		super(type(self), self).__init__()
		self.img_train_list = img_train_list
	def __len__(self):
		return len(self.img_train_list)
	def __getitem__(self, index):
		raise Exception("method hasn't been implemented")


class TestDataset(Dataset):
	def __init__(self, img_test_list):
		super(type(self), self).__init__()
		self.img_test_list = img_test_list
	def __len__(self):
		return len(self.img_test_list)
	def __getitem__(self, index):
		raise Exception("method hasn't been implemented")
