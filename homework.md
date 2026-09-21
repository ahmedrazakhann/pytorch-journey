1. Practice matrices on paper.
2. Learn and practice eigen values. # https://www.youtube.com/watch?v=h8sg_XBp6VA (look this resource, if it can help)
3. Read this resource: (https://horace.io/brrr_intro.html)

## Extra-curriculum

- There have been several iterations and tweaks to the Vision Transformer since its original release and the most concise and best performing (as of July 2022) can be viewed in [_Better plain ViT baselines for ImageNet-1k_](https://arxiv.org/abs/2205.01580). Despite of the upgrades, we stuck with replicating a "vanilla Vision Transformer" in this notebook because if you understand the structure of the original, you can bridge to different iterations.
- The [`vit-pytorch` repository on GitHub by lucidrains](https://github.com/lucidrains/vit-pytorch) is one of the most extensive resources of different ViT architectures implemented in PyTorch. It's a phenomenal reference and one I used often to create the materials we've been through in this chapter.
- PyTorch have their [own implementation of the ViT architecture on GitHub](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py), it's used as the basis of the pretrained ViT models in `torchvision.models`.
- Jay Alammar has fantastic illustrations and explanations on his blog of the [attention mechanism](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) (the foundation of Transformer models) and [Transformer models](https://jalammar.github.io/illustrated-transformer/).
- Adrish Dey has a fantastic [write up of Layer Normalization](https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1) (a main component of the ViT architecture) can help neural network training.
- The self-attention (and multi-head self-attention) mechanism is at the heart of the ViT architecture as well as many other Transformer architectures, it was originally introduced in the [_Attention is all you need_](https://arxiv.org/abs/1706.03762) paper.
- Yannic Kilcher's YouTube channel is a sensational resource for visual paper walkthroughs, you can see his videos for the following papers:
  - [Attention is all you need](https://www.youtube.com/watch?v=iDulhoQ2pro) (the paper that introduced the Transformer architecture).
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://youtu.be/TrdevFK_am4) (the paper that introduced the ViT architecture).
