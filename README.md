## LLM from scrath

### Quick start
* simple train with only Shakespeare txt
    - python bigram_model.py --mode simple_train

* pre train with openweb txt and then run fine tuning with Shakespeare's txt
    - python bigram_model.py --mode train

* run inference with trained checkpoint
    - python bigram_model.py --mode infer

### Data preparation
* extrac openweb txt tar file and save them as txt files
    - cd pretrain_data_prepare
    - python prepare.py

### Reference
* gpt-dev.ipynb
    - it is a scratch board to learn encode/decode, embedding, self-attention

* bigram.py
    - simple LM with just encode/decode, embedding

* bigram_v2.py
    - Full model (decoder only) for LLM
    ```
    python bigram_v2.py
    ```