import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as init
import constants
import torch.nn.functional as F

class Question_model(nn.Module):
	def __init__(self,embedding_dimensions,vocab_size,lstm_hidden_size,drop_p=0):
		super(Question_model,self).__init__()
		self.embedding=nn.Embedding(vocab_size,embedding_dimensions,padding_idx=0)# to make embeddings for token at index 0 to be 0
		nn.init.xavier_uniform(self.embedding.weight)
		self.drop=nn.Dropout(drop_p)
		self.tanh=nn.Tanh()
		self.lstm=nn.LSTM(embedding_dimensions,lstm_hidden_size,bidirectional=True)
		nn.init.xavier_uniform(self.lstm.weight_ih_l0)
		nn.init.xavier_uniform(self.lstm.weight_hh_l0)
		self.lstm.bias_ih_l0.data.zero_()
		self.lstm.bias_hh_l0.data.zero_()

	def forward(self,ques_indexed,ques_len):
		embeds=self.embedding(ques_indexed)
		tanh=self.tanh(embeds)
		#print(tanh.shape)
		pack_sequence=pack_padded_sequence(tanh,ques_len,batch_first=True) 
		#print(pack_sequence)
		'''
		  since in a batch all ques of different lengths but ques_indexed of size max_ques 
		  lenghth with extra index filled with zeros.so this  packs the padded sequence which can be 
		  directly fed to lstms.
		  Note that now the output of lstm total_hidden is also a packed sequence so if needed we may have to
		  unpack it
		'''
		total_hidden,(current_hidden,final_cell_state)=self.lstm(pack_sequence)
		# final_cell_state has dimention (num_layers * num_directions, batch, hidden_size)
		# so removing the extra dimention
		print("heya",final_cell_state)
		print("heyab",final_cell_state.size())
		#final_cell_state=final_cell_state.squeeze(0) 
		return final_cell_state[-1]

class Attention_model(nn.Module):
	def __init__(self,image_feature_size,question_feature_size,common_feature_size,n_glimpse,drop=0):
		super(Attention_model,self).__init__()
		self.conv=nn.Conv2d(image_feature_size,common_feature_size,1) # 1*1 kernel to reduce dimensions, remember inception net
		self.linear=nn.Linear(question_feature_size,common_feature_size)
		self.drop=nn.Dropout(drop)
		self.conv2=nn.Conv2d(common_feature_size,n_glimpse,1)
		self.relu=nn.ReLU(inplace=True)

	def forward(self,image_features,question_feature):
		img_f=self.conv(self.drop(image_features))
		print("hsada",img_f.size())
		print("gagan",question_feature.size())
		ques_f=self.linear(question_feature)
		print("choman",ques_f.size())
		batch_size,common_feature_size=ques_f.size()
		print("batch_size",batch_size,common_feature_size)
		ques_f=ques_f.view(batch_size,common_feature_size,*([1,1])).expand_as(img_f) 
		# converting a feature vecture to a veature map by replicating vector at each element to make it 
		# the same shape as img features so as to concatenate and create probablitiy distribution
		attention=self.relu(ques_f+img_f)
		attention=self.conv2(self.drop(attention))
		# now find the probablity distributionover the 2 glimpses of image
		# more than one glimpse is required for focussing on diiferent parts of image based on different context of 
		# words in large questions

		#here after img_f has shape(batch_size,common_feature_size,14,14)
		# ques_f has shape(batch_size,common_feature_size,14,14)
		# attention has shape (batch_size,2,14,14)
		batch_size,n_glimpse,img_x,img_y=attention.shape
		feature_size=image_features.size(1)
		prob=F.softmax(attention.view(batch_size,n_glimpse,-1),dim=2) # flatten the tensor to take softmax over entire image 
		img_feat=image_features.view(batch_size,1,feature_size,img_x*img_y).expand_as(torch.zeros([batch_size,1,feature_size,img_x*img_y]))
		attention = prob.view(batch_size, n_glimpse, 1, img_y*img_x)
		# .expand_as(torch.zeros([batch_size,1,feature_size,img_x*img_y]))
		# print(attention.shape,img_feat.shape)
		weighted_features = img_feat * attention
		# sum over only the spatial dimension
		weighted_mean = weighted_features.sum(dim=3).view(batch_size,-1)
    	# now shape (batch_size,n_glimpse*common_feature_size)
		return prob.view(batch_size,n_glimpse,img_x,img_y), weighted_mean

class Classifier_model(nn.Module):
	def __init__(self,input_features_size,hidden_features_size,output_features_size,drop=0.0):
		super(Classifier_model,self).__init__()
		self.drop=nn.Dropout(drop)
		self.lin1=nn.Linear(input_features_size,hidden_features_size)
		self.relu=nn.ReLU()
		self.lin2=nn.Linear(hidden_features_size,output_features_size)
		self.logsoftmax=nn.LogSoftmax()
	def forward(self,inputs):
		lin=self.lin1(self.drop(inputs))
		relu=self.relu(lin)
		lin2=self.lin2(self.drop(relu))

		return self.logsoftmax(lin2)

class Network(nn.Module):
	def __init__(self, vocab_size):
		super(Network, self).__init__()
		question_feature_size = 1024
		img_features_size =constants.img_features_size
		n_glimpses = 2

		self.ques = Question_model(
			embedding_dimensions=300,
			vocab_size=vocab_size,
			lstm_hidden_size=question_feature_size,
			drop_p=0.5
		)
		self.attention = Attention_model(
			image_feature_size=img_features_size,
			question_feature_size=question_feature_size,
			common_feature_size=512,
			n_glimpse=2,
			drop=0.5
		)
		self.classifier = Classifier_model(
		    input_features_size=n_glimpses*img_features_size+question_feature_size,
		    hidden_features_size=1024,
		    output_features_size=3000,
		    drop=0.0
		)

		for m in self.modules():
		    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
		        init.xavier_uniform(m.weight)
		        if m.bias is not None:
		            m.bias.data.zero_()

	def forward(self, img_features,ques_indexed, ques_len):
		q_encoded = self.ques(ques_indexed, list(ques_len.data))
		img_encoded = img_features / (img_features.norm(p=2, dim=1, keepdim=True).expand_as(img_features) + 1e-8)
		probs, attend = self.attention(img_encoded, q_encoded)
		concatenated = torch.cat([attend, q_encoded], dim=1)
		answer = self.classifier(concatenated)
		return answer,probs

		



