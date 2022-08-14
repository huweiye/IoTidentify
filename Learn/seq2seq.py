# 代码来源：https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html?highlight=seq2seq
#模型图片：https://pytorch.org/tutorials/_images/seq2seq_ts.png
#把data目录下的数据删了所以它跑不起来,只要下载回来就能跑起来

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

device = torch.device("cpu")

MAX_LENGTH = 10  # Maximum sentence length

# 默认的词向量
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    '''使用Voc对象来包含从单词到索引的映射，以及词汇表中的单词总数'''

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # 统计SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # 统计默认的令牌
        for word in keep_words:
            self.addWord(word)


# 小写并删除非字母字符
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 接受一个单词的句子并返回相应的单词索引序列
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

#定义编码器
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        '''

        :param input_size: 输入的embedding的维数
        :param hidden_size: GRU的输出维数
        :param embedding:???????????
        :param n_layers: GRU层数/深度
        :param dropout:缺省值为0,表示不使用dropout层;若为1，则除最后一层外，其它层的输出都会加上dropout层
        :param bidirectional:是否是双向RNN
        '''
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                      dropout=(0 if n_layers == 1 else dropout), bidirectional=True)#注意！没有！设置batch_first

    def forward(self, input_seq, input_lengths, hidden=None):
        '''
                    :param self:
                    :param input_seq:
                    :param input_lengths:
                    :param hidden:GRU的初始参数，不用管
                    :return:
                    '''
        # 将单词索引转换为向量
        embedded = self.embedding(input_seq)
        '''
                    关于pack_padded_sequence和pad_packed_sequence的作用见：
                    https://blog.csdn.net/m0_46483236/article/details/124136437
                    '''

        print("Shape of GRU's input:",embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,
                                                         input_lengths)  # padding后的输入序列先经过nn.utils.rnn.pack_padded_sequence，这样会得到一个PackedSequence类型的object，
                                                                        # 返回值可以直接传给RNN,RNN的源码中的forward函数里上来就是判断输入是否是PackedSequence的实例，进而采取不同的操作，如果是则输出也是该类型。

        outputs, hidden = self.gru(packed, hidden)  # 传进去的是packed，返回也是packed，所以记得要unpack

        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            outputs)  # 再经过nn.utils.rnn.pad_packed_sequence，也就是对经过RNN后的输出重新进行padding操作，得到正常的每个样本都相等时刻数目的序列

        print("output size() of bidirectional gru=", outputs.size())  # (seq_len, batch, hidden_size * 2)
        # 对双向RNN来说，每个样本的每个时间步的输出有两个，一个正向一个反向，他们被简单的拼接起来，所以第三个维度长度是hidden_size * 2

        # 将双向GRU的输出结果简单求和，向量对应元素相加
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # 将双向RNN的两个outputs

        # 返回输出以及最终的隐藏状态
        return outputs, hidden

'''
这里的global attention method详见：
    https://blog.csdn.net/m0epNwstYk4/article/details/81073986

解码器的attention子模块:和上面的EncoderRNN一样也是一个神经网络模块
    输入：指定attention的align method:其实就是指定如何计算权重向量at的方法，对生成t时刻单词的decoder_t来说，它除了自己按照标准RNN产生一个ht，还需要一个ct,这个ct就是全体encoder每个时刻的输出乘以相应的权重后再求和得到的向量：weighted summation
    见图：https://img-blog.csdnimg.cn/img_convert/5d3cd0463bd1c4188a1e872727bc6b8c.png
    其中权重向量at的每个元素都是每个encoder的输出和decoder_t的输出ht调用method生成的，这里有三种：'dot', 'general', 'concat'
    
    输出：根据给定的方法生成当前decoder_t的权重向量at
'''
class Attn(nn.Module):
    def __init__(self,  method, hidden_size):
        '''
        :param method: 指定attention的align方法
        :param hidden_size:RNN的标准输出——隐藏状态的维数
        '''
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    #以下三个函数用不同的方法生成当前decoder对应的权重向量
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # 根据给定的方法生成当前decoder_t的权重向量at
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # 转置 max_seq_length and batch_size dimensions
        attn_energies = attn_energies.t()#t():?

        # 返回softmax归一化概率分数（增加维度）
        return F.softmax(attn_energies, dim=1).unsqueeze(1)#unsqueeze():?

#定义解码器
'''
Similarly to the EncoderRNN, we use the torch.nn.GRU module for our decoder’s RNN. However,This time,we use a unidirectional GRU. 
It is important to note that unlike the encoder, we will feed the decoder RNN one word at a time. 
-> 这句话的意思是说,解码的时候，一次只输入给解码器一个单词embedding，只有在得到当前解码器的输出才能送入解码器进行下一次训练，而编码器的RNN是一次送入一整个句子
We start by getting the embedding of the current word and applying a dropout. 
Next, we forward the embedding and the last hidden state to the GRU,then the GRU obtain a current GRU output and hidden state. 
    We then use our Attn module as a layer to obtain the attention weights to get attended encoder output. We use this attended encoder output as our context tensor. 
    From here, we use a linear layer and softmax normalization to select the next word in the output sequence.
'''
#解码器
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_method, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        '''

        :param attn_method:Attention层align的方法，['dot', 'general', 'concat']
        :param embedding:??????????
        :param hidden_size:GRU的输出维数
        :param output_size:最终输出维数
        :param n_layers:GRU层数
        :param dropout:神经元宕机的概率，Dropout只能用在训练部分而不能用在测试部分
        '''
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)#nn.Dropout(p = 0.3) # 表示每个神经元有0.3的可能性不被激活

        # 当前时刻GRU的输入是上一时刻GRU的输出，所以input_size=hidden_size,hidden_size也是hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))#注意没有开启batch_first，所以time_seq为第0维
        #生成Attn模块对象，传进去attention align的方法和GRU的输出维数
        self.attn = Attn(attn_method, hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)#

        self.out = nn.Linear(hidden_size, output_size)#最终输出层

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time，即一个时间步执行一次forward
        # Get embedding of current input word，获取当前输入单词对应的embedding
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

#定义整个编码器和解码器的运行过程
'''
编码：对输入序列进行编码很简单：只需将整个序列张量及其对应的长度向量转发给编码器。
需要注意的是，该模块一次只处理一个输入序列，而不是一批序列。因此，当常数1用于声明张量大小时，这对应于batch_size的大小为1。

解码：要对给定的解码器输出进行解码，我们必须通过我们的解码器模型迭代地运行前向传递，该模型输出对应于每个单词成为下一个单词的概率的softmax分数。
每次通过解码器后，我们贪婪地将softmax概率最高的单词附加到解码单词列表中，并且使用这个词作为下一次迭代的解码器输入。
如果解码单词列表已达到最大长度，或者如果预测单词是EOS_标记，则解码过程终止。
'''
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)#对一整个句子来说，编码器只调用这一次
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_softmaxscores = torch.zeros([0], device=self._device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):#迭代地调用解码器，一次输入一个word token，它来自上一个时刻解码器的输出，而不是向编码器那样一次输入一整个句子
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_softmaxscores = torch.cat((all_softmaxscores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_softmaxscores  #输出编码器预测的句子

#预测输入
'''
evaluate函数获取一个规范化的字符串句子，将其处理为相应单词索引的张量（batch_size大小为1），
并将该张量传递给名为searcher的GreedySearchDecoder实例，以处理编码/解码过程。

searcher返回输出单词索引向量和对应于每个解码单词标记的softmax分数的分数张量。最后一步是使用voc.index2word将每个单词索引转换回其字符串表示形式。
'''
def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


# Evaluate inputs from user input (stdin)
def evaluateInput(searcher, voc):
    input_sentence = ''
    while(True):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break#用户输入q则退出
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

# Normalize input sentence and call evaluate()
def evaluateExample(sentence, searcher, voc):
    print("> " + sentence)
    # Normalize sentence
    input_sentence = normalizeString(sentence)
    # Evaluate sentence
    output_words = evaluate(searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))



save_dir = os.path.join("../data", "save")
corpus_name = "cornell movie-dialogs corpus"

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# If you're loading your own model
# Set checkpoint to load from
checkpoint_iter = 4000
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                             '{}_checkpoint.tar'.format(checkpoint_iter))

# If you're loading the hosted model
loadFilename = 'data/4000_checkpoint.tar'

# Load model
# Force CPU device options (to match tensors in this tutorial)
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc = Voc(corpus_name)
voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
# Load trained model params
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()
print('Models built and ready to go!')



### Compile the whole greedy search model to TorchScript model
# Create artificial inputs
test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words).to(device)
test_seq_length = torch.LongTensor([test_seq.size()[0]]).to(device)
# Trace the model
traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))

### Convert decoder model
# Create and generate artificial inputs
test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
# Trace the model
traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))

### Initialize searcher module by wrapping ``torch.jit.script`` call
scripted_searcher = torch.jit.script(GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers))


# Use appropriate device
scripted_searcher.to(device)
# Set dropout layers to eval mode
scripted_searcher.eval()

# Evaluate examples
sentences = ["hello", "what's up?", "who are you?", "where am I?", "where are you from?"]
for s in sentences:
    evaluateExample(s, scripted_searcher, voc)

# Evaluate your input
#evaluateInput(traced_encoder, traced_decoder, scripted_searcher, voc)