#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
model = transformers.TFAutoModel.from_pretrained("ufal/eleczech-lc-small", output_hidden_states=True)

dataset = [
    "Podmínkou koexistence jedince druhu Homo sapiens a společenství druhu Canis lupus je sjednocení akustické signální soustavy.",
    "U závodů na zpracování obilí, řízených mytologickými bytostmi je poměrně nízká produktivita práce vyvážena naprostou spolehlivostí.",
    "Vodomilní obratlovci nepatrných rozměrů nejsou ničím jiným, než vodomilnými obratlovci.",
]

print("Textual tokenization")
print([tokenizer.tokenize(sentence) for sentence in dataset])

print("Char - subword - word mapping")
encoded = tokenizer(dataset[0])
print("Token 2: {}".format(encoded.token_to_chars(2)))
print("Word 1: {}".format(encoded.word_to_tokens(1)))
print("Char 12: {}".format(encoded.char_to_token(12)))
for i in range(50):
    print("Char {}: {}".format(i, encoded.char_to_token(i)))
print(type(encoded.token_to_chars(60)), encoded.token_to_chars(60) is None)
print(encoded.token_to_chars(0))
print(encoded.token_to_chars(2).start)

print("Tokenization to IDs")
batch = [tokenizer.encode(sentence) for sentence in dataset]
print(batch)

max_length = max(len(sentence) for sentence in batch)
print(max_length)
batch_ids = np.zeros([len(batch), max_length], dtype=np.int32)
batch_masks = np.zeros([len(batch), max_length], dtype=np.int32)
for i in range(len(batch)):
    batch_ids[i, :len(batch[i])] = batch[i]
    batch_masks[i, :len(batch[i])] = 1

result = model(batch_ids, attention_mask=batch_masks)
print("last_hidden_state: shape {}".format(result.last_hidden_state.shape))
print("hidden_state: shapes", *("{}".format(hidden_state.shape) for hidden_state in result.hidden_states))
#print(result)
