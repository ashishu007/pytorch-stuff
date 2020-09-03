# pytorch-stuff
This repository contains a bunch of different things I tried for learning PyTorch.

1. `anno-gpt` folder contains some code for GPT2, more specifically:
    - writing GPT2 from scratch; and
        - **Credits**: [The Annotated GPT-2](https://amaarora.github.io/2020/02/18/annotatedGPT2.html)
    - fine-tuning pre-trained GPT2 using `huggingface`
        - **Credits**: [Ian Pointer's Book](https://snappishproductions.com/blog/2020/03/01/chapter-9.5-text-generation-with-gpt-2-and-only-pytorch.html.html)

2. `dgl-stuff` folder contains some materials for learning Graph Neural Networks using DGL library:
    - **Credits**: [DGL's tutorials](https://docs.dgl.ai/en/0.4.x/tutorials/basics/1_first.html)

3. `g2s` folder contains some code for creating a Graph-to-Sequence model using [Graph Attention Netwroks](https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html#put-everything-together) as encoder and [GPT2](https://amaarora.github.io/2020/02/18/annotatedGPT2.html) as decoder
    - not completed yet

4. `s2s` folder contains some code for learning seq2seq using Transfomers:
    - **Credits**: [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
