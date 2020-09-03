# pytorch-stuff
This repository contains a bunch of different things I tried for learning PyTorch.

1. `annogpt` folder contains some code for GPT2, more specifically:
    - writing GPT2 from scratch; and
        - **Credits**: [The Annotated GPT-2](https://amaarora.github.io/2020/02/18/annotatedGPT2.html)
    - fine-tuning pre-trained GPT2 using `huggingface`
        - **Credits**: [Ian Pointer's Book](https://snappishproductions.com/blog/2020/03/01/chapter-9.5-text-generation-with-gpt-2-and-only-pytorch.html.html)

2. `dglstuff` folder contains some materials for learning Graph Neural Networks using DGL library:
    - **Credits**: [DGL's tutorials](https://docs.dgl.ai/en/0.4.x/tutorials/basics/1_first.html)

3. `g2s` folder contains some code for creating a Graph-to-Sequence model using [Relational Graph Convolutional Netwroks](https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/4_rgcn.html) as encoder and [GPT2](https://amaarora.github.io/2020/02/18/annotatedGPT2.html) as decoder
    - still working on this
    - a lot of code here is taken from [this repository](https://github.com/zhaochaocs/DualEnc/)
        - current SOTA on WebNLG dataset

4. `s2s` folder contains some code for learning seq2seq using Transfomers:
    - **Credits**: [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
