import os
import gzip
import pickle
from collections import defaultdict
from tqdm import tqdm


# note: some classical special token
MASK_TOKEN = "<mask>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
BLANK_TOKEN = "<blank>"


class DYTokenizer:
    def __init__(self, vocab_file=None, vocab_size=None, special_token=True, max_len=None):
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.special_token = special_token
        self.token_idx = defaultdict(int)
        self.idx_token = defaultdict(str)
        self.max_len = max_len
        self._initialize()
    
    def _initialize(self):
        if self.special_token:
            self.mask_token = MASK_TOKEN
            self.unk_token = UNK_TOKEN
            self.pad_token = PAD_TOKEN
            self.bos_token = BOS_TOKEN
            self.eos_token = EOS_TOKEN
            self.blank_token = BLANK_TOKEN
        else:
            self.mask_token = None
            self.unk_token = None
            self.pad_token = None
            self.bos_token = None
            self.eos_token = None
            self.blank_token = None
        
        assert not self.vocab_file or not self.vocab_size 
        if not self.vocab_file or not os.path.isfile(self.vocab_file):
            print("Vocab_file is not foind, using unk token initialize tokenizer.")
            for i in range(self.vocab_size):
                self.idx_token[i] = UNK_TOKEN
            self.token_idx[UNK_TOKEN] = self.vocab_size - 1
        else:
            with open(self.vocab_file, "r") as f:
                for line in f:
                    content = line.strip().split()
                    idx = len(self.token_idx)
                    self.token_idx[content[0]] = idx
                    self.idx_token[idx] = content[0]
            print(f"Vocab size : {len(self.token_idx)}")
            
    # def token2idx(self, batch_sample, max_len=300):
    #     """
    #     Transfer text to idx sequence.
    #     paremeter:
    #         -content: list[list], include 'batch_size' setences 
    #     """
    #     max_len = min()
    def token2idx(self, token):
        return self.token_idx[token]
    
    def idx2token(self, idx):
        return self.idx_token[idx]
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(name_or_path={self.vocab_file}, vocab_size={self.vocab_size}, max_len={self.max_len})")



if __name__ == '__main__':
    tokenizer = DYTokenizer(vocab_file="/mnt/data/duyaoo/dy/data/phoenix-2014-T/text_vocab.txt")
    print(tokenizer)