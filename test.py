from data_loader import encode_que
import numpy as np 
import torch
from image_features_generator import RESNET
from PIL import Image
import os
import constants
import json
from model_definition_old import Network
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
def test(filename,que):
	use_cuda=True
	use_cuda = use_cuda and torch.cuda.is_available()
	path=filename
	orig_img=Image.open(path).convert('RGB')

	with open('vocab.json') as vocab:
		vocab=json.load(vocab)

	transform=torchvision.transforms.Compose([
        transforms.Scale(constants.input_img_size),
        transforms.CenterCrop(constants.input_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),])
	img=transform(orig_img)
	img=img.view(1,img.size(0),img.size(1),img.size(2))
	resnet=RESNET()
	if use_cuda:
		resnet.cuda()

	resnet.eval() # making the model in eval mode cz we are using pretrained model
	if use_cuda:
		img=img.cuda()
	out = resnet(img)
	feature_img=out.data.cpu()

	question=que
	question = question.lower()[:-1]
	question=question.split(' ')

	ques,ques_len=encode_que(question,constants.max_question_length,vocab)
	ques_len=torch.tensor(ques_len)
	ques_len=ques_len.view(1,1)
	ques=ques.view(1,ques.size(0))
	model=Network(len(vocab['question'])+1)
	if use_cuda:
		model=model.cuda()
	
	checkpoint = torch.load('2018-11-22 11_52_26.680564.pth', map_location=lambda storage, loc: storage)
	model.load_state_dict(checkpoint['weights'])

	ques = Variable(ques)
	feature_img = Variable(feature_img)
	ques_len = Variable(ques_len)
	if use_cuda:
		feature_img = Variable(feature_img.cuda(async=True))
		ques = Variable(ques.cuda(async=True))
		ques_len = Variable(ques_len.cuda(async=True))

	output,probs = model(feature_img,ques,ques_len)
	probs=torch.nn.functional.upsample(probs,size=[constants.input_img_size,constants.input_img_size])
	probs=probs.detach().cpu().numpy()

	prob,index = torch.topk(output,5,dim=1)
	index=index.data[0].cpu().numpy()
	prob=prob.data[0].cpu().numpy()
	finalprobs = np.exp(prob)
	keys=np.array(list(vocab['answer'].keys()))
	values=vocab['answer'].values()
	
	
	test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((constants.input_img_size,constants.input_img_size))
        # transforms.ToTensor()
    ])
	prob0=np.reshape(probs[0,0,:,:],[448,448,1])
	prob1=np.reshape(probs[0,1,:,:],[448,448,1])
	prob0=gaussian_filter(prob0,10)
	prob1=gaussian_filter(prob1,10)
	prob1=prob1/np.max(prob1)
	prob0=prob0/np.max(prob0)
	# print(orig_img.size)
	test_img=np.array(orig_img.resize([448,448]))
	# print(test_img.shape)
	# test_img=(np.array(test_transform(np.array(orig_img)))/255).astype(np.float32)
	# print(test_img.shape)
	# print(np.max(prob0),np.max(prob1),np.max(test_img),np.max(test_img*prob1).astype(np.uint8))
	# plt.figure(),plt.imshow(prob0[:,:,0],cmap='gray')
	# plt.figure(),plt.imshow(prob1[:,:,0],cmap='gray')
	# plt.imshow(test_img)
	# plt.figure(),plt.imshow((test_img/255+prob0)/np.max(test_img/255+prob0))
	# plt.savefig(str(path)+"attn1.png")
	# plt.figure(),plt.imshow((test_img/255+prob1)/np.max(test_img/255+prob1))
	# plt.savefig(str(path)+"attn2.png")
	# # plt.figure(),plt.imshow((test_img*prob0).astype(np.uint8))
	# # plt.figure(),plt.imshow((test_img*prob1).astype(np.uint8))
	# plt.show()
	attn1 = (test_img/255+prob0)/np.max(test_img/255+prob0)
	attn2 = (test_img/255+prob1)/np.max(test_img/255+prob1)
	return keys[index],finalprobs,(255*attn1).astype(int),(255*attn2).astype(int)
		