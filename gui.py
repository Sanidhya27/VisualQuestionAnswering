from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from test import test
import matplotlib.pyplot as plt
from PIL import ImageTk, Image

class Root(Tk):
	def __init__(self):
		super(Root, self).__init__()
		self.title("VQA") 
		self.attributes('-zoomed', True)
		self.labelFrame = ttk.LabelFrame(self, text = "Open a file")
		self.labelFrame.place(x = 60, y = 10)
		self.rowconfigure(9, {'minsize': 30})
		self.columnconfigure(9, {'minsize': 30})
		#self.geometry('{}x{}'.format(1600, 900))
		self.filename = None
		self.question = None
		self.panel1 = Label(self)
		self.panel2 = Label(self)
		self.panel3 = Label(self)
		self.label = Label(self)

		self.initUI()
		self.buttons()

	def showImage(self,path,pos_x,pos_y):
	    img = Image.open(path)
	    img = img.resize((330,250), Image.ANTIALIAS)
	    img = ImageTk.PhotoImage(img)
	    panel = Label(self,image = img)
	    panel.image = img
	    panel.place(x = pos_x, y= pos_y)

	def numImage(self,data,pos_x,pos_y,index):
	    img = Image.fromarray(data.astype('uint8'), 'RGB')
	    img = img.resize((330,250), Image.ANTIALIAS)
	    img = ImageTk.PhotoImage(img)
	    panel = Label(text = "Glimpse " + str(index))
	    panel.place(x = pos_x, y = pos_y - 50)
	    panel = Label(self,image = img)
	    panel.image = img
	    panel.place(x = pos_x, y= pos_y)

	def buttons(self):
		self.button = ttk.Button(self.labelFrame, text = "Browse images", command = self.fileDialog)
		self.button.grid(column = 1, row = 1)

	def clickMe(self):
		self.panel1.destroy()
		self.panel2.destroy()
		self.panel3.destroy()
		if self.filename != None:
			self.question = self.name.get()
			keys,probs,attn1,attn2 = test(self.filename,self.question)
			self.panel1 = Label(text = probs)
			self.panel1.place(x = 800, y = 200)
			self.panel2 = Label(text = keys)
			self.panel2.place(x = 800, y = 150)
			self.panel3 = Label(text = "Top 5 answers with probabilities are:")
			self.panel3.place(x = 800, y = 100)
			self.numImage(attn1,60,400,1)
			self.numImage(attn2,800,400,2)
		

	def initUI(self):
		self.name = StringVar()
		panel = Label(self, text = "Enter the question")
		panel.place(x = 800, y = 10)
		textbox = Entry(self, width = 50, textvariable = self.name)
		textbox.place(x = 800, y = 50)
		self.button = Button(self, text = "Submit", command = self.clickMe, width = 10)
		self.button.place(x = 1080, y = 10)

	def fileDialog(self):
		self.label.destroy()
		self.filename = filedialog.askopenfilename(initialdir = "/home/sanidhya/Downloads/project", title = "Select a file", filetypes = (("jpeg","*.jpg"),("All Files", "*.*")))
		self.label = Label(text = self.filename)
		self.label.place(x = 60, y = 50)
		self.showImage(self.filename,60,100)

root = Root()
root.mainloop()