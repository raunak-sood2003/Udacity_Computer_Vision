import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        batch = features.size()[0]
        captions = captions[:, :-1]
        captions = self.embedding(captions)
        features = features.unsqueeze(1)
        
        lstm_input = torch.cat((features, captions), 1)
        
        #Giving me an error
        h_0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        
        #Works when I use None
        out, _ = self.lstm(lstm_input, None)
        out = self.fc(out)
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_ids = []
        
        for i in range(max_len+1):
            out, _ = self.lstm(inputs, states)
            out = self.fc(out)
            
            preds = torch.argmax(out, dim = 2)
            output_ids.append(int(preds.item()))
            
            inputs = self.embedding(preds)
            
        
        return output_ids
