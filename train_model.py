import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
from tqdm import tqdm
import data_loader as data
import model_definition_old as model
import constants 
import csv
total_iter = 0

def batch_accuracy(output,true):
	_,index = output.max(dim=1, keepdim=True)
	agreeing = true.gather(dim=1, index=index)
	return (agreeing * 0.3).clamp(max=1)

def update_learning_rate(optimizer, iteration):
    lr = constants.initial_lr * 0.5**(float(iteration) / 50000)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def step(net, loader, optimizer, epoch, train = False):
	# log_softmax = nn.LogSoftmax().cuda()
	answers = []
	accuracies=[]
	losses=[]
	if train == True:
		net.train()
	else:		
		net.eval()

	for img,que,ans,que_len,index in tqdm(loader):
		img = Variable(img.cuda(async=True))
		que = Variable(que.cuda(async=True))
		ans = Variable(ans.cuda(async=True))
		que_len = Variable(que_len.cuda(async=True))
		output,probs = net(img,que,que_len)
		loss = -(output*ans/10).sum(dim=1).mean()
		accuracy = batch_accuracy(output.data, ans.data).cpu()
		accuracies.append(accuracy.mean().data.cpu().numpy())
		losses.append(loss.data.cpu().numpy())
		if train:
			global total_iter
			update_learning_rate(optimizer, total_iter)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_iter += 1
			
			if(total_iter%100==0 or total_iter==1): 
				print('Train')
				print('epochs:[%d/%d]\t iterations: [%d]\t Loss: %.4f\t accuracy: %.4f\t'
		                  % (epoch, constants.epochs,total_iter,
		                     loss.item(),np.mean(accuracies)))
	if not train:
		print('Val')
		print('epochs:[%d/%d]\t iterations: [%d]\t Loss: %.4f\t accuracy: %.4f\t'
	                  % (epoch, constants.epochs,total_iter,
	                     loss.item(),np.mean(accuracies)))
			
	
	return np.mean(accuracies),np.mean(losses)
def main():
	train_loader = data.data_loader(train=True)
	val_loader = data.data_loader(train=False)
	net = (model.Network(len(train_loader.dataset.vocab['question'])+1)).cuda()
	optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
	name = datetime.now()
	name = os.path.join('logs', '{}.pth'.format(name))
	max_val_acc=0
	print("Model is runnning")
	train_losses=[]
	val_losses=[]
	train_accuracies=[]
	val_accuracies=[]
	for i in range(constants.epochs):
		train_acc,train_loss = step(net,train_loader,optimizer,i,train = True)
		val_acc,val_loss = step(net,val_loader,optimizer,i)
		train_losses.append(train_loss)
		train_accuracies.append(train_acc)
		val_losses.append(val_loss)
		val_accuracies.append(val_acc)
		if(val_acc>max_val_acc):
			max_val_acc=val_acc
			global total_iter
			results = {
				'name': datetime.now(),
				'weights': net.state_dict(),
				'epoch': i,
				'iterations': total_iter,
				'val_acc': val_acc
				}
			print('saving_model')
			torch.save(results, name)
	# np.save('train_losses.npy',np.array(train_losses))
	# np.save('val_losses.npy',np.array(val_losses))
	# np.save('train_acc.npy',np.array(train_accuracies))
	# np.save('val_acc.npy',np.array(val_accuracies))
	rows = zip(train_losses,train_accuracies,val_losses,val_accuracies)


	with open('meta_data.csv', "w") as f:
		writer = csv.writer(f)
		for row in rows:
			writer.writerow(row)

if __name__ == '__main__':
    main()