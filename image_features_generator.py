import torch
import h5py
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import constants
from tqdm import tqdm
from PIL import Image
import os
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import resnet as caffe_resnet

use_cuda=True
use_cuda = use_cuda and torch.cuda.is_available()
print('using cuda:',use_cuda)

class RESNET(nn.Module):
	def __init__(self):
		super(RESNET,self).__init__()
		self.model=nn.Sequential(*list(models.resnet152(pretrained=True).children())[:-2]) #removing the FC layer
		#self.model=nn.Sequential(*list(caffe_resnet.resnet152(pretrained=True).children())[:-2]) #removing the FC layer
	def forward(self,input_img):
		output=self.model(input_img)
		return output

class img_dataset(data.Dataset):

	def __init__(self,path,transform=None):
		super(img_dataset,self).__init__()
		self.path=path
		self.transform=transform
		self.id_to_filename={}
		for filename in os.listdir(self.path):
			if(filename.endswith('.jpg')):
				id_img=int((filename.split('_')[-1]).split('.')[0])
				self.id_to_filename[id_img]=filename
		self.sorted_ids=sorted(self.id_to_filename.keys())
	
	def __getitem__(self,index):
		id_img=self.sorted_ids[index]
		path=os.path.join(self.path,self.id_to_filename[id_img])
		img=Image.open(path).convert('RGB')
		if (self.transform is not None):
			img=self.transform(img)
		return id_img,img

	def __len__(self):
		return len(self.sorted_ids)

def get_images_loader(mode):

	if(mode=='train'):
		path='/home/sanidhya/dataset/train2014'

	elif(mode=='val'):
		path='/home/sanidhya/dataset/val2014'

	transform=torchvision.transforms.Compose([
        transforms.Scale(constants.input_img_size),
        transforms.CenterCrop(constants.input_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),])
	dataset=img_dataset(path,transform)
	print(len(dataset))
	data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=constants.preprocess_batch_size,
        num_workers=constants.num_workers,
        shuffle=False,
        pin_memory=True,)
	return data_loader

def main():
	resnet=RESNET()
	if use_cuda:
		resnet.cuda()

	resnet.eval() # making the model in eval mode cz we are using pretrained model

	val_loader=get_images_loader('val')
	train_loader=get_images_loader('train')
	print(len(train_loader.dataset),len(train_loader))
	features_shape = (
        len(train_loader.dataset),
        constants.img_features_size,
        constants.output_img_size,
        constants.output_img_size
    )
	with h5py.File('pretrained_train_img_features', libver='latest') as pretrained_train_imgs:
		features_train = pretrained_train_imgs.create_dataset('features', shape=features_shape, dtype='float16')
		image_ids_train = pretrained_train_imgs.create_dataset('ids', shape=(len(train_loader.dataset),), dtype='int32')

		i = j = 0
		for ids, imgs in tqdm(train_loader):
			if use_cuda:
				imgs=imgs.cuda()

			out = resnet(imgs)
			j = i + imgs.size(0)
			features_train[i:j] = out.data.cpu().numpy().astype('float16')
			image_ids_train[i:j] = ids.numpy().astype('int32')
			i = j
	features_shape = (
        len(val_loader.dataset),
        constants.img_features_size,
        constants.output_img_size,
        constants.output_img_size
    )
	with h5py.File('pretrained_val_imgs_features', libver='latest') as pretrained_val_imgs:
		features_val = pretrained_val_imgs.create_dataset('features', shape=features_shape, dtype='float16')
		image_ids_val = pretrained_val_imgs.create_dataset('ids', shape=(len(val_loader.dataset),), dtype='int32')

		i = j = 0
		for ids, imgs in tqdm(val_loader):
			if use_cuda:
				imgs=imgs.cuda()

			out = resnet(imgs)
			j = i + imgs.size(0)
			features_val[i:j] = out.data.cpu().numpy().astype('float16')
			image_ids_val[i:j] = ids.numpy().astype('int32')
			i = j

			
if __name__ == '__main__':
	main()
		