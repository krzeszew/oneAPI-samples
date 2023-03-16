#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================
# Copyright © 2023 Intel Corporation
# 
# SPDX-License-Identifier: MIT
# =============================================================


# # Interactive chat based on DialoGPT model using Intel® Extension for PyTorch* Quantization
# 
# This code sample shows usage of DiloGPT model as interactive chat with Intel Extension for PyTorch INT8 quantization.
# 
# ## DialoGPT
# 
# DialoGPT is a model based on GPT-2 architecture proposed by Microsoft in 2019. It's goal was to create open-domain chatbots capable of producing natural responses to a variety of conversational topics.

# Let's start with importing all neccessery packages.

# In[ ]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import warnings
warnings.filterwarnings('ignore')


# ## Model and tokenizer loading
# 
# The first implemented function is loading tokenizer and model. 
# 
# Function input is link to the pre-trained model. In this sample we are using `microsoft/DialoGPT-large` from HuggingFace. This is also default parameter for this function. Of course, you can use also `microsoft/DialoGPT-medium` or `microsoft/DialoGPT-samll` models. Especially if you have limited resources. 

# In[ ]:


def load_tokenizer_and_model(model="microsoft/DialoGPT-large"):
    """
    Load tokenizer and model instance for some specific DialoGPT model.
    """
    # Initialize tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    
    # Return tokenizer and model
    return tokenizer, model


# ## INT8 Dynamic Quantization
# 
# **Quantization** is a systematic reduction of the precision of all or several layers within the model. This means that we turn a higher-precision type, such as the FP32 (32 bits) most commonly used in Deep Learning, into a lower-precision type, such as FP16 (16 bits) or INT8 (8 bits). 
# 
# With type reduction, it is possible to effectively reduce the size of the model and also faster inference. That means:
# 
# * lower memory badwidth, 
# * lower storage, 
# * higer performance with minimum to zero accuracy loss. 
# 
# This is especially important, with large models such as those based on the Transformers architecture, like BERT or used in this sample GPT. 
# 
# We can distinguish 2 types of quantization:
# 
# * static - requres an additional pass over a dataset to work, only actications do calibration,
# * dynamic - mulriplies input values by the scale factor, then rounds the result to the nearest, the scale factor for activatuons is determined dynamically based on the data range observed in runtime.
# 
# In this sample we are using **the dynamic quantization**.

# In[ ]:


from intel_extension_for_pytorch.quantization import prepare, convert
import intel_extension_for_pytorch as ipex

def quantize_model(tokenizer, model):
    """
    Adding IPEX dynamic qulatization to the model
    """
    # Evaluate model
    model.eval()
    
    print("Quantization in progres...")
    
    # Prepare example outputs for the model
    question, text = "What is SYCL?", "SYCL is an industry-driven standard, developed by Kronos Group and announced in March 2014."
    inputs = tokenizer(question, text, return_tensors="pt")
    jit_inputs  = tuple((inputs['input_ids']))
    
    # Create configuratgion for dynamic quantization
    qconfig = ipex.quantization.default_dynamic_qconfig
    
    # Optimize model
    model = ipex.optimize(model)
    
    # Prepare model for quantization using prevously prepared partameters
    prepared_model = prepare(model, qconfig, example_inputs=jit_inputs, inplace=False)
    
    # Convert types in model
    converted_model = convert(prepared_model)
    
    return tokenizer, converted_model


# ## Response generation 
# 
# Response generation in DialoGPT architecture based on **encoder-decoder** model. It means that first we need to *encode input sentence*, to later on be able to *decode it* generating resonse.
# 
# As the model based on transformers architecture they have known issue of copying things. To avoid reprtition in chat responces we used Top-K sampling and Top-p sampling.
# 
# **Top-K sampling** filters the K most likely next words and redistributes the probability mass among only those K next words. **Top-p sampling**, rather than selecting only the most likely K words, selects the smallest possible set of words whose cumulative probability exceeds the probability p. The probability mass is then redistributed among the words in this set. As a result, the size of the set of words can be dynamically increased and decreased based on the probability distribution of the next word.

# In[ ]:


def generate_response(tokenizer, model, chat_round, chat_history_ids):
    """
    Generate a response to some user input.
    """
    # Encode user input and End-of-String (EOS) token
    new_input_ids = tokenizer.encode(input(">> You:") + tokenizer.eos_token, return_tensors='pt')
    
    # Append tokens to chat history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_round > 0 else new_input_ids
    
    # Generate response given maximum chat lenght history of 2000 tokens
    chat_history_ids = model.generate(
        bot_input_ids,
        do_sample=True, 
        max_length=2000,
        top_k=50, 
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Print response
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    
    # Return the chat history ids
    return chat_history_ids


# The next step is to prepare a function that allows interactive conversation for `n` rounds. This means that we will use the previously prepared `generate_response` function n-times.

# In[ ]:


def chat_for_n_rounds(tokenizer, model, n=5):
    """
    Chat with chatbot for n rounds (n = 5 by default)
    """

    # Initialize history variable
    chat_history_ids = None

    # Chat for n rounds
    for chat_round in range(n):
        chat_history_ids = generate_response(tokenizer, model, chat_round, chat_history_ids)


# Now, it is time to use implemented functions - initializing the model and adding INT8 dynamic quantization.

# In[ ]:


# Initialize tokenizer and model
tokenizer, model = load_tokenizer_and_model()

# Adding ipex quantization to the model
tokenizer, model = quantize_model(tokenizer, model)


# Let's play with the model by 5 rounds. 

# In[ ]:


chat_for_n_rounds(tokenizer, model, 5)


# DialoGPT by Microsoft is another conversational chatbot that everyone can use. 
# 
# Based on this architecture, we created an interactive chat in this sample. The use of top-k and top-p allowed us to avoid some of the repetition in the chat answers. Furthermore, the addition of dynamic INT8 quantization reduced memory usage.

# In[ ]:


print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")

