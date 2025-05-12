This repository consists of 3 projects :

1. **LSTM-Based Sequence to sequence Model for Text Summarization** Implemented a BERT-based Seq2Seq model for text summarization. It uses pretrained BERT embeddings in a custom encoder-decoder architecture. Input articles and summaries are tokenized, padded, and fed to the model for training. The generated summaries are evaluated using ROUGE metrics to assess n-gram overlap with reference summaries. The project includes training, inference, and evaluation pipelines.

2. **Embedding and Visualization of Tokens via Rest API** This project explores word embedding techniques, specifically TF-IDF, and uses dimensionality reduction for visualization.  It includes a REST API (built with Flask) to serve embedding data.  Additionally, it has a web application that allows users to input words and see their embeddings and nearest neighbors.

3. **Text Preprocessing functionality deployed through rest API based web surface**: Implemented basic nlp operations like tokenization , stemming , lemmitization , POS Tag , NER in web interface via Rest Api
