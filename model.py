# English to French Translation
# DOC : https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
from torch.cuda import is_available
from torch import nn, optim, device, zeros, tensor, long, no_grad, load, save
from torch.nn import functional as F
import datetime
import time
from numpy import array
import unicodedata
import random
from matplotlib import pyplot as plt
import os

if is_available():
    device=device("cuda")
else:
    device=device("cpu")

logging_mode=0
teacher_forcing_ratio=0.5 # It is concept of using real target outputs as each next input instead of using decoder's guess as next input.

class logger:
    def __init__(self, log_file_name="log.log"):
        self.log_fd=open(log_file_name,"w")
    def log(self,txt):
        if logging_mode==0:
            return
        self.log_fd.write(f"{datetime.datetime.now().strftime('[%d-%m-%Y %H:%M:%S]')} {txt}\n")
    def close(self):
        self.log_fd.close()

log_handler=logger()

class PreProcess:
    def __init__(self,data_filename="data/eng-fra.txt"):
        log_handler.log(f"Reading the datafile {data_filename}")
        with open(data_filename,encoding="utf-8") as datafile:
            self.raw_data_sentences=datafile.readlines()
        log_handler.log("Initializing the Preprocessing class variables.")
        self.SOS=0  # Start Of Sentence token       
        self.EOS=1  # End Of Sentence token
        self.idx2wrd_ENG={self.SOS : "SOS", self.EOS : "EOS"}
        self.idx2wrd_FRA={self.SOS : "SOS", self.EOS : "EOS"}
        self.n_words_ENG=2
        self.n_words_FRA=2
        self.distinct_ENG=[]
        self.distinct_FRA=[]
        self.X=[]
        self.Y=[]
    def segment(self):
        log_handler.log("Segmenting the dataset to extract the English and French data from the datafile.")
        for i in self.raw_data_sentences:
            i=i.strip()
            sent=i.split('	')
            #eng=''.join(asc for asc in unicodedata.normalize('NFD',self.removeStopWords(sent[0])) if unicodedata.category(asc)!='Mn')
            #fra=''.join(asc for asc in unicodedata.normalize('NFD',self.removeStopWords(sent[1])) if unicodedata.category(asc)!='Mn')
            eng=self.removeStopWords(sent[0])
            fra=self.removeStopWords(sent[1])
            temp=eng.split(' ')
            for j in temp:
                j=j.lower()
                if j not in self.distinct_ENG:
                    self.idx2wrd_ENG[self.n_words_ENG]=j
                    self.n_words_ENG+=1
                    self.distinct_ENG.append(j)
            temp=fra.split(' ')
            for j in temp:
                j=j.lower()
                if j not in self.distinct_FRA:
                    self.idx2wrd_FRA[self.n_words_FRA]=j
                    self.n_words_FRA+=1
                    self.distinct_FRA.append(j)
        log_handler.log(f"Segmentation ended with #English Words = {self.n_words_ENG} and #French Words = {self.n_words_FRA}.")
    def removeStopWords(self,snt):
        #log_handler.log("Removing the stop words from the sentences.")
        stop_words=['.',',',';','?','!','\n','%','\\','\'']
        for i in stop_words:
            snt=snt.replace(i,'')
        #log_handler.log("Stop word removal completed.")
        return snt
    def sequencing(self):
        log_handler.log("Sequence generation started.")
        wrd2idx_ENG={self.idx2wrd_ENG[j] : j for j in self.idx2wrd_ENG.keys()}
        wrd2idx_FRA={self.idx2wrd_FRA[j] : j for j in self.idx2wrd_FRA.keys()}
        for i in self.raw_data_sentences:
            sent=i.split('	')
            eng=self.removeStopWords(sent[0])
            fra=self.removeStopWords(sent[1])
            temp=eng.split(' ')
            arr=[]
            for j in temp:
                j=j.lower()
                arr.append(wrd2idx_ENG[j])
            arr.append(self.EOS)
            self.X.append(arr)
            temp=fra.split(' ')
            arr=[]
            for j in temp:
                j=j.lower()
                arr.append(wrd2idx_FRA[j])
            arr.append(self.EOS)
            self.Y.append(arr)
        #self.X=tensor(self.X, dtype=long, device=device).view(-1,1)
        #self.Y=tensor(self.Y, dtype=long, device=device).view(-1,1)
        log_handler.log("Sequence generation completed.")  
    def getTensorPairs(self,n):
        done=[]
        pair=[]
        for i in range(n):
            idx=random.randint(0, len(self.X)-1)
            if idx not in done:
                pair.append((tensor(self.X[idx],dtype=long,device=device).view(-1,1),tensor(self.Y[idx],dtype=long,device=device).view(-1,1)))
                done.append(idx)
        return pair
    def getSequences(self, X):
        wrd2idx_ENG={self.idx2wrd_ENG[j] : j for j in self.idx2wrd_ENG.keys()}
        X=self.removeStopWords(X)
        temp=X.split(' ')
        arr=[]
        for i in temp:
            i=i.lower()
            arr.append(wrd2idx_ENG[i])
        return tensor(arr,dtype=long,device=device).view(-1,1)
    def decode(self,X):
        return self.idx2wrd_FRA[X]

preprocessing_engine=PreProcess()
preprocessing_engine.segment()
preprocessing_engine.sequencing()


class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Encoder,self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size, hidden_size)
        self.gru=nn.GRU(hidden_size, hidden_size)
    def forward(self,X,hidden=None):
        if hidden==None:
            hidden=zeros(1,1,self.hidden_size,device=device)
        embedded=self.embedding(X).view(1,1,-1)
        output=embedded
        output, hidden=self.gru(output,hidden)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(Decoder,self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(output_size, hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.logits=nn.LogSoftmax(dim=1)
    def forward(self,X,hidden=None):
        if hidden==None:
            hidden=zeros(1,1,self.hidden_size,device=device)
        output=self.embedding(X).view(1,1,-1)
        output=F.relu(output)
        output, hidden = self.gru(output, hidden)
        output=self.logits(self.out(output[0]))
        return output, hidden

def train(X,Y,criterion,maxlength=100):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_hidden=None

    input_length=X.size(0)
    target_length=Y.size(0)
    encoder_outputs=zeros(maxlength,encoder.hidden_size, device=device)

    loss=0
    for inp in range(input_length):
        output, encoder_hidden=encoder(X[inp],encoder_hidden)
        encoder_outputs[inp]=output[0,0]
    
    decoder_input=tensor([[0]], device=device)
    decoder_hidden=encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for inp in range(target_length):
            output, decoder_hidden=decoder(decoder_input,decoder_hidden)
            loss+=criterion(output,Y[inp])
            decoder_input=Y[inp]
    else:
        for inp in range(target_length-1):
            output, decoder_hidden=decoder(decoder_input,decoder_hidden)
            top_v, top_i=output.topk(1)
            decoder_input=top_i.squeeze().detach()
            loss+=criterion(output, Y[inp])
            if decoder_input.item() == 1:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainFn(iterations):
    pairs=preprocessing_engine.getTensorPairs(iterations)
    loss_aggregate=[]
    criterion=nn.NLLLoss()

    for iter in range(1, iterations+1):
        start=time.time()
        training_pair=pairs[(iter)%len(pairs)]
        input_tensor=training_pair[0]
        target_tensor=training_pair[1]
        loss=train(input_tensor,target_tensor,criterion)
        loss_aggregate.append(loss)
        end=time.time()
        if iter%1000==0:
            print(f"[{iter}/{iterations}] - Loss={loss} Time Taken={round(end - start,4)}")
            save({
                "model" : encoder.state_dict(),
                "optimizer" : encoder_optimizer.state_dict() 
            },"encoder.tar.gz")
            save({
                "model" : decoder.state_dict(),
                "optimizer" : decoder_optimizer.state_dict() 
            },"decoder.tar.gz")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(loss_aggregate)
    plt.savefig("trainning.png")

hidden_size=256
learning_rate=0.01
encoder=Encoder(preprocessing_engine.n_words_ENG,hidden_size).to(device)
decoder=Decoder(hidden_size, preprocessing_engine.n_words_FRA).to(device)
encoder_optimizer = optim.SGD(encoder.parameters(), learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), learning_rate)

if 'encoder.tar.gz' in os.listdir('.') and 'decoder.tar.gz' in os.listdir('.'):
    encoder.load_state_dict(load("encoder.tar.gz",map_location=device)['model'])
    decoder.load_state_dict(load("decoder.tar.gz",map_location=device)['model'])
    encoder_optimizer.load_state_dict(load("encoder.tar.gz",map_location=device)['optimizer'])
    decoder_optimizer.load_state_dict(load("decoder.tar.gz",map_location=device)['optimizer'])

else:
    trainFn(70000)

def test():
    inp=input(">")
    while(inp not in ['q','Q','QUit','quit','exit','Exit']):
        X=preprocessing_engine.getSequences(inp)
        with no_grad():
            encoder_hidden=None
            encoder_outputs=zeros(100,encoder.hidden_size,device=device)
            for i in range(X.size()[0]):
                output, encoder_hidden=encoder(X[i],encoder_hidden)
                encoder_outputs[i] += output[0,0]
            decoder_input=tensor([[0]],device=device)
            decoder_hidden=encoder_hidden
            decoded_words=[]

            for i in range(100):
                output, decoder_hidden=decoder(decoder_input,decoder_hidden)
                top_v, top_i=output.data.topk(1)
                if top_i.item() == 1:
                    decoded_words.append("<EOS>")
                    break
                else:
                    decoded_words.append(preprocessing_engine.decode(top_i.item()))
                decoder_input=top_i.squeeze().detach()
            print(' '.join(decoded_words))
        inp=input(">")




if __name__=='__main__':
    test()




