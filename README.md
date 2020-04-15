# DS8008-final-project
Replicating the abstractive text summarization results from the paper "Text Summarization with Pretrained Encoders" with a MUCH smaller dataset.

**Link to paper**: https://arxiv.org/pdf/1908.08345v2.pdf

**Abstract of linked paper:**  
Bidirectional Encoder Representations from Transformers (BERT; Devlin et al. 2019) represents the latest incarnation of pretrained language models which have recently advanced a wide range of natural language processing tasks. In this paper, we showcase how BERT
can be usefully applied in text summarization and propose a general framework for both extractive and abstractive models. We introduce a novel document-level encoder based on BERT which is able to express the semantics of a document and obtain representations for its sentences. Our extractive model is built on top of this encoder by stacking several intersentence Transformer layers. For abstractive summarization, we propose a new fine-tuning schedule which adopts different optimizers for the encoder and the decoder as a means of alleviating the mismatch between the two (the former is pretrained while the latter is not). We also demonstrate that a two-staged fine-tuning approach can further boost the quality of the generated summaries. Experiments on three datasets show that our model achieves stateof-the-art results across the board in both extractive and abstractive settings.

**Note on analysis**

All analysis for this project used the Hugging Face library in Python.

Link to Hugging Face github page:

https://github.com/huggingface

Link to BERT Abstractive Summarization folder in the Hugging Face library:

https://github.com/huggingface/transformers/tree/master/examples/summarization/bertabs
