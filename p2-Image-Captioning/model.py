import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    '''Code is referred from:
    1. https://github.com/sunsided/image-captioning/blob/develop/model.py
    2. https://github.com/vmelan/CVND-udacity/blob/master/P2_Image_Captioning/model.py
    3. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py
    '''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 0 else 0,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):

        # output (num_layers, batch_size, hidden_size)
        return (torch.zeros((1, batch_size, self.hidden_size), device=device),
                torch.zeros((1, batch_size, self.hidden_size), device=device))


    def forward(self, features, captions):
        batch_size = features.size(0)

        self.hidden = self.init_hidden(batch_size)

        cap_no_end = captions[:, :-1] # (batch_size, caption_length -1)
        cap_embed = self.embed(cap_no_end) # output (batch_size, caption_length -1, embed_size)

        # stack features and captions
        features = features.unsqueeze(1) # output (batch_size, 1, embed_size)
        cap_embed = torch.cat((features, cap_embed), dim=1) # output (batch_size, caption_length, embed_size)

        lstm_out, self.hidden = self.lstm(cap_embed, self.hidden)
        # lstm_out -> (batch_size, caption_length, hidden_size)

        outputs = self.fc(lstm_out) # output (batch_size, caption_length, vocab_size)

        return outputs

    def sample(self, inputs, max_len=20):
        """

        :param inputs (tensor): (batch_size=1, caption_length=1, embed_size) features input
        :param max_len (int): caption maximal length
        :return predictions (list): list of integer
        """
        batch_size = inputs.size(0) # inputs -> (1, caption_length=1, embed_size)
        self.hidden = self.init_hidden(batch_size) # hidden -> (num_layers, batch_size=1, hidden_size)

        predictions = []

        for _ in range(max_len):
            lstm_out, self.hidden = self.lstm(inputs, self.hidden) # lstm_out -> (1, caption_length=1, hidden_size)
            outputs = self.fc(lstm_out) # outputs (1,caption_length=1, vocab_size)
            outputs = outputs.squeeze(1) # outputs (batch_size=1, vocab_size)
            _, max_idx = torch.max(outputs, dim=1) # max_idx -> (1,)
            predictions.append(max_idx.cpu().numpy()[0].item()) # return python integer

            if max_idx == 1:
                break

            inputs = self.embed(max_idx) # output (batch_size=1, embed_size)
            inputs = inputs.unsqueeze(1) # output (batch_size=1, caption_length=1, embed_size)

        return predictions

    def beam_search(self, features, k=3, end_id=1, max_len=20):
        """Code is referred from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py
        
        Accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len).
        Beam Search is implemented in sampling with default K=3.

        :param features (tensor): (batch_size=1, caption_length=1, embed_size)
        :param k (int): size of beam search
        :param end_id (int): <end> token id
        :param max_len (int): maximal sentence length
        :return:
            final_seq (list): list of tokenized word for best sentence
        """

        seq_scores, seqs = self.forward_features(features, k)
        step = 1  # seqs length counter
        # seq_scores -> (k,1)
        # seqs -> (k,1)

        complete_seq = list()
        complete_score = list()

        while True:
            #print('step:', step)
            scores = self.forward_word(seq_scores, seqs)
            # scores -> (k, vocab_size)
            # h -> (k, hidden_size)
            # c -> (k, hidden_size)

            top_scores, top_ids = scores.view(-1).topk(k, dim=0, largest=True, sorted=True)
            # top_scores -> (k,)
            # top_ids -> (k,)

            pre_seq_ids = top_ids / self.vocab_size  # (k,) -> it's integer tensor, thus division will ground down to floor
            next_word_ids = top_ids % self.vocab_size  # (k,)
            seqs = torch.cat((seqs[pre_seq_ids], next_word_ids.unsqueeze(1)), dim=1)  # (k, step+1) -> new seqs
            left_ids = [ind for ind, next_word in enumerate(next_word_ids)
                        if next_word != end_id]  # (s,) -> incompleted sentences
            complete_ids = set(range(len(next_word_ids))) - set(left_ids)
            complete_ids = list(complete_ids)

            if len(complete_ids) > 0:
                complete_seq.extend(seqs[complete_ids].tolist())  # list of lists
                #print('complete_seq:', complete_seq)
                complete_score.extend(top_scores[complete_ids].cpu().detach().numpy())  # list of numbers
                #print('complete_score:', complete_score)

            k -= len(complete_ids)  # update to remaining beam search size (k -> s)

            if k == 0:
                break

            seqs = seqs[left_ids]  # (s, steps)
            seq_scores = top_scores[left_ids].unsqueeze(1)  # (s, 1)

            h = self.hidden[0][:, pre_seq_ids, :] # hidden state for new set of seqs
            h = h[:, left_ids, :] # (1, s, hidden_size) -> hidden state for left sequences

            c = self.hidden[1][:, pre_seq_ids, :] # hidden state for new set of seqs
            c = c[:, left_ids, :] # (1, s, hidden_size) -> hidden state for left sequences

            self.hidden = (h,c)

            step += 1
            if step >= max_len:
                break

        final_id = complete_score.index(max(complete_score))  # .index() picks up the idx of the max value
        final_seq = complete_seq[final_id]

        return final_seq


    def forward_features(self, features, k):
        """

        :param features (tensor): (1, caption_length=1, embed_size)
        :param k (int): size of beam search
        :return:
            seq_scores (tensor): (k,1)
            seqs (tensor): (k,1)
            h (tensor): (num_layers=1, k, hidden_size)
            c (tensor): (num_layers=1, k, hidden_size)
        """

        batch_size = features.size()[0] # inputs -> (1, caption_length=1, embed_size)
        self.hidden = self.init_hidden(batch_size) # hidden -> (num_layers, batch_size=1, hidden_size)

        lstm_out, self.hidden = self.lstm(features, self.hidden)
        # lstm_out -> (1, 1, hidden_size)
        # hidden -> (num_layers, batch_size=1, hidden_size) for current time step

        preds = self.fc(lstm_out) # (1, 1, vocab_size)
        scores = F.log_softmax(preds, dim=2)
        top_scores, top_ids = scores.topk(k, dim=2, largest=True, sorted=True) # both shape (1,1,k)

        # treat the beam search sequences as batch_size k
        seq_scores = top_scores.view(-1, 1) # (k, 1)
        seqs = top_ids.view(-1, 1) # (k, 1)

        h = torch.cat([self.hidden[0] for _ in range(k)], dim=1) # (1, batch_size=k, hidden_size)
        c = torch.cat([self.hidden[1] for _ in range(k)], dim=1) # (1, batch_size=k, hidden_size)

        self.hidden = (h,c)

        return seq_scores, seqs

    def forward_word(self, seq_scores, seqs):
        """

        :param seq_scores (tensor): (k,1)
        :param seqs (tensor): (k, steps)
        :param h (tensor): (k, hidden_size)
        :param c (tensor): (k, hidden_size)
        :return:
            scores (tensor): (k, vocab_size)
            h (tensor): (num_layers=1, k, hidden_size)
            c (tensor): (num_layers=1, k, hidden_size)
        """

        word = seqs[:, -1] # (k,)

        w_embed = self.embed(word) # (batch_size=k, embed_size)
        w_embed = w_embed.unsqueeze(1) # (batch_size=k, caption_length=1, embed_size)
        lstm_out, self.hidden = self.lstm(w_embed, self.hidden)
        # lstm_out -> (k, 1, hidden_size)
        # h -> (1, k, hidden_size)
        # c -> (1, k, hidden_size)

        preds = self.fc(lstm_out) # (k, 1, vocab_size)
        scores = F.log_softmax(preds, dim=2) # (k, 1, vocab_size)
        scores = scores.squeeze(1) # (k, vocab_size)
        scores = seq_scores.expand_as(scores) + scores

        return scores













