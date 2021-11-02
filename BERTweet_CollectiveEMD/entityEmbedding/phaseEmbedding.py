import pandas as pd
import numpy as np
import re


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

# import pandas as pd
# import numpy as np

learning_rate = 0.0001

class PhraseEmbedding(nn.Module):

    def __init__(self,input_size,output_size,device):
        super(PhraseEmbedding, self).__init__()
        self.print_once=True
        self.dense_layer = nn.Linear(input_size,output_size)
        self.non_linear_layer = nn.Tanh()
        self.cosine_layer = nn.CosineSimilarity(dim=0)
        self.device = device
        return

    def encode(self, input_embedding):

        # print(input_embedding.size())
        input_sentence_embedding = input_embedding.squeeze(0)
        # print(input_sentence_embedding.size())
        # print('-----')

        # Max Pool
        # max_pooled_embedding = torch.max(input_sentence_embedding,dim=0)

        #Average Pool
        average_pooled_embedding = torch.mean(input_sentence_embedding,dim=0).to(device=self.device)
        # print(average_pooled_embedding.size())

        x = self.dense_layer(average_pooled_embedding)
        # print(x.size())

        out = self.non_linear_layer(x)
        # print(out.size())
        return out

    def forward(self, input_tuple):
        # print(len(input_tuple))
        input_source = input_tuple[0]
        input_target = input_tuple[1]

        output_source = self.encode(input_source)
        output_target = self.encode(input_target)

        similarity = self.cosine_layer(output_source, output_target)
        # print(similarity)
        return similarity

    def getEmbedding(self, input_embeddings):
        with torch.no_grad():
            return self.encode(input_embeddings)