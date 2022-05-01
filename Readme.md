# Usage
python model.py

# Working
The script will initialize the encoder decoder models and train them on the provided dataset. The code written takes in english and french samples from data applied preprocessing on them to tokenize and seqence encode them. At last the model checkpoints are saved in form of gunzip files. If gunzip files alredy exists then you will be given with a prompt in which if you enter an english sentence then it will be converted into equivalent french sentences. 
The extent of meaningfulness of the generated translations depend upon the training epochs chosen (As POC, I found that epochs=70K is sufficient).

# Future work
I want to add self attention to the decoder model and check if training epochs can be reduced or not mainintaining efficacy of the translation.
