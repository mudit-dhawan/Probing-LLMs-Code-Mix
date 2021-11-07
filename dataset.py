import config
import torch

class LIDdataset():
    def __init__(self, texts, lidtags, tokenizer=config.TOKENIZER):
        # texts = [["aap","ka","health","kaisa", "hai", "?"], ["accha", "hun"]]
        # lidtags = [["hi","hi","en","hi","hi","en"], ["hi","hi"]]
        self.texts = texts
        self.lidtags = lidtags
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # If Pre-Training -> Since we already performed BPE, we may not need it again.
        # If Fine-Tuning -> Use built-in tokenizer
        data = self.texts[idx]
        tags = self.lidtags[idx]
        
        # print(data, tags)
        text = [text.strip() for text in eval(data)]
        lidtag = ['hi' if (senttag.strip() == 'Hin') else 'en' for senttag in eval(tags)]
        
        # print(text, lidtag)
        
        ids = []
        langtags = []

        for i, wrd in enumerate(text):
            inputs = self.tokenizer.encode(wrd).ids
            
            # For applying BPE
            input_len = len(inputs)
            ids.extend(inputs)
            langtags.extend([lidtag[i]]*input_len)
        
        # print(langtags)
        
        # print([(langtag, type(langtag)) for langtag in langtags])
        
        ids = ids[:config.MAX_LEN-2]
        langtags = langtags[:config.MAX_LEN-2]

        # Mask your sentence here
        # Returns token_ids with masked ids and masking labels (-100 for non-masks)
        # and vocab_index for masked words
        ids, masking_labels = mask_sentence(ids, self.tokenizer)
        
        # Reference : https://huggingface.co/transformers/model_doc/xlm.html#transformers.XLMTokenizer.build_inputs_with_special_tokens
        ids = [0] + ids + [1]
        masking_labels = [-100] + masking_labels + [-100] # For <s> & </s> on either side.
        langtags =  [self.tokenizer.lang2id["en"]] + [self.tokenizer.lang2id[langtag] for langtag in langtags] + [self.tokenizer.lang2id["en"]]
        # Additional "en" on either side is for <s> & </s>
        
        mask =  [1]*len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        langtags = langtags + ([0] * padding_len)
        masking_labels = masking_labels + ([-100] * padding_len)
        """
        `attention_mask` (torch.FloatTensor of shape (batch_size, sequence_length), optional) â€“
        Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        1 for tokens that are not masked,
        0 for tokens that are maked.

        `token_type_ids` (torch.LongTensor of shape (batch_size, sequence_length), optional) â€“
        Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:
            0 corresponds to a sentence A token,
            1 corresponds to a sentence B token.
        """

        """
        For labels,
        
        Ref: https://huggingface.co/transformers/model_doc/xlm.html#transformers.XLMWithLMHeadModel.forward
        labels (torch.LongTensor of shape (batch_size, sequence_length), optional) â€“ 
        Labels for language modeling. Note that the labels are shifted inside the model, i.e. 
        you can set labels = input_ids 
        Indices are selected in [-100, 0, ..., config.vocab_size].
        All labels set to -100 are ignored (masked), the loss is only computed for labels
        in [0, ..., config.vocab_size]
        """

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(masking_labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "langs": torch.tensor(langtags, dtype=torch.long)
        }

import random
def mask_sentence(token_ids, tokenizer, style='bert'):
    """
    Reference: https://github.com/huggingface/transformers/blob/f9cde97b313c3218e1b29ea73a42414dfefadb40/examples/lm_finetuning/simple_lm_finetuning.py#L267
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param token_ids: list of str (token-ids), tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here for <mask> to id)
    :return: (list of token-ids that are input, list of int), masked tokens and related labels for LM prediction
    Replace some with <mask>, some with random words.
    """
    output_label = []
    # mask_positions = [] # For storing the position where words are changed
    
    for i, token_id in enumerate(token_ids):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                token_ids[i] = tokenizer.mask_token_id

            # 10% randomly change token to random token
            # elif prob < 0.9:
            #     tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            
            output_label.append(token_id)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-100)
    
    return token_ids, output_label 