import json
import os
import torch
import torch.utils.data as data
import h5py
import re
from collections import Counter
import itertools
import constants

train_path = 'pretrained_train_img_features'
val_path = 'pretrained_val_img_features'
train_questions_path = 'data/OpenEnded_mscoco_train2014_questions.json'
train_answers_path = 'data/mscoco_train2014_annotations.json'
val_questions_path = 'data/OpenEnded_mscoco_val2014_questions.json'
val_answers_path = 'data/mscoco_val2014_annotations.json'

comma = re.compile(r'(\d)(,)(\d)')
period = re.compile(r'(?!<=\d)(\.)(?!\d)')
punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
punctuation = re.compile(r'([{}])'.format(re.escape(punctuation_chars)))
punctuation_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(punctuation_chars))

def collate(batch):
	batch.sort(key=lambda x: x[-2], reverse=True)
	return data.dataloader.default_collate(batch)

def extract_answers(iterable):
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    most_common = counter.most_common(constants.max_answers)
    most_common = (t for t, c in most_common)
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens)}
    return vocab

def extract_questions(iterable):
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    most_common = counter.keys()
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=1)}
    return vocab

def encode_que(question,max_question_length,vocab):
	encoded = torch.zeros(max_question_length).long()
	for i,token in enumerate(question):
		encoded[i] = vocab['question'].get(token,0)
	return encoded, len(question)

def encode_ans(answer,vocab):
	encoded = torch.zeros(len(vocab['answer']))
	for a in answer:
		if vocab['answer'].get(a) != None:
			encoded[vocab['answer'].get(a)] += 1
	return encoded

def punctuation_process(s):
	if punctuation.search(s) is None:
            return s
	s = punctuation_space.sub('', s)
	if re.search(comma, s) is not None:
            s = s.replace(',', '')
	s = punctuation.sub(' ', s)
	s = period.sub('', s)
	return s.strip()

def coco_index(path):
	with h5py.File(path,'r') as file:
		idx = file['ids'][()]
		coco_to_index = {id: i for i, id in enumerate(idx)}
	return coco_to_index

def tokenize_ques(questions_json):
	questions = [q['question'] for q in questions_json['questions']]
	for question in questions:
		question = question.lower()[:-1]
		yield question.split(' ')

def tokenize_ans(answers_json):
	answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]  
	for answer in answers:
		yield list(map(punctuation_process, answer)) 

class VQADATASET(data.Dataset):
	def __init__(self,image_path,questions_path,answers_path,train=False):
		super(VQADATASET, self).__init__()
		with open(questions_path, 'r') as file:
			questions_json = json.load(file)
		with open(answers_path, 'r') as file:
			answers_json = json.load(file)  
		self.image_path = image_path
		self.id_to_index = coco_index(image_path) 
		self.ids = [q['image_id'] for q in questions_json['questions']]

		questions = list(tokenize_ques(questions_json))
		answers = list(tokenize_ans(answers_json))
		question_vocab = extract_questions(questions)
		answer_vocab = extract_answers(answers)
		if train:
			self.vocab = {
		    	'question': question_vocab,
				'answer': answer_vocab,
			}
			with open('vocab.json', 'w') as file:
				json.dump(self.vocab, file)
		else:
			with open('vocab.json', 'r') as file:
				self.vocab=json.load(file)


		self.max_question_length = max(map(len, questions))  

		self.questions = [encode_que(q,self.max_question_length,self.vocab) for q in questions]
		#print("huhu",self.questions)
		self.answers = [encode_ans(a,self.vocab) for a in answers]

	def load_img(self,image_id):
		file = h5py.File(self.image_path,'r')
		index = self.id_to_index[image_id]
		dataset = file['features']
		img = dataset[index].astype('float32')
		return torch.from_numpy(img)

	def __getitem__(self,index):
		img_idx = self.ids[index]
		image = self.load_img(img_idx)	
		questions,que_len = self.questions[index]
		answers = self.answers[index]
		return image, questions, answers, que_len, index

	def __len__(self):
		return len(self.questions)

def data_loader(train=True):
	if train:
		image_path=train_path
		questions_path=train_questions_path
		answers_path=train_answers_path
		dataset = VQADATASET(image_path,questions_path,answers_path,train=True)
	else:
		image_path=val_path
		questions_path=val_questions_path
		answers_path=val_answers_path
		dataset = VQADATASET(image_path,questions_path,answers_path,train=False)

	

	loader = data.DataLoader(
					dataset,
					batch_size = constants.batch_size,
					pin_memory = True,
					collate_fn = collate
					)
	return loader