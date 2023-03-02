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
    def __init__(self, vocab_file, vocab_size=None, special_token=True):
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.special_token = special_token
        self.token_idx = defaultdict(int)
        self.idx_token = defaultdict(int)
    
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
    
    