# RAG - Retrieval Augmented Generation

In this we are goin to build a basic pipeline for RAG and run it locally.

It will be from PDF ingestion to "chat with the PDF" feature.

I this i have used all open source tools easily available.

!["This is a flowchart describing a simple local retrieval-augmented generation (RAG) workflow for document processing and embedding creation, followed by search and answer functionality. The process begins with a collection of documents, such as PDFs or a 1200-page nutrition textbook, which are preprocessed into smaller chunks, for example, groups of 10 sentences each. These chunks are used as context for the Large Language Model (LLM). A cool person (potentially the user) asks a query such as "What are the macronutrients? And what do they do?" This query is then transformed by an embedding model into a numerical representation using sentence transformers or other options from Hugging Face, which are stored in a torch.tensor format for efficiency, especially with large numbers of embeddings (around 100k+). For extremely large datasets, a vector database/index may be used. The numerical query and relevant document passages are processed on a local GPU, specifically an RTX 4090. The LLM generates output based on the context related to the query, which can be interacted with through an optional chat web app interface. All of this processing happens on a local GPU. The flowchart includes icons for documents, processing steps, and hardware, with arrows indicating the flow from document collection to user interaction with the generated text and resources."](images/simple-local-rag-workflow-flowchart.png)


## Getting started

- If you don't have a local NVIDIA GPU with VRAM of 5GB or above, you can 
- - You can run on CPU but it will be time consuming
- - You can work on Google COLAB and select GPU there to access

## Prerequisites

- Comfortable writing Python code. 
- 1-2 beginner machine learning/deep learning courses.
- Familiarity with PyTorch

## Clone Repo

```
git clone https://github.com/NiketGirdhar22/RAG_from_scratch.git
```

```
cd RAG_from_scratch
```

### Create  and activate enviornment

- Creation

```
python -m venv venv
```

- Activation

Linux/MacOS
```
source venv/bin/activate
```

Windows
```
.\venv\Scripts\activate
```

### Install requirements

```
pip install -r requirements.txt
```

Note: Installation in torch may be have to done manually, install `torch` manually (`torch` 2.1.1+ is required for newer versions of attention for faster inference) with CUDA, see: https://pytorch.org/get-started/locally/

On Windows used:

```
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Launch notebook

VS Code:

```
code .
```

Jupyter Notebook

```
jupyter notebook
```

## Key terms

| Term | Description |
| ----- | ----- | 
| **Token** | A sub-word piece of text. For example, "hello, world!" could be split into ["hello", ",", "world", "!"]. A token can be a whole word, part of a word or group of punctuation characters. 1 token ~= 4 characters in English, 100 tokens ~= 75 words. Text gets broken into tokens before being passed to an LLM. |
| **Embedding** | A learned numerical representation of a piece of data. For example, a sentence of text could be represented by a vector with  768 values. Similar pieces of text (in meaning) will ideally have similar values. |
| **Embedding model** | A model designed to accept input data and output a numerical representation. For example, a text embedding model may take in 384  tokens of text and turn it into a vector of size 768. An embedding model can and often is different to an LLM model. |
| **Similarity search/vector search** | Similarity search/vector search aims to find two vectors which are close together in high-demensional space. For example,  two pieces of similar text passed through an embedding model should have a high similarity score, whereas two pieces of text about  different topics will have a lower similarity score. Common similarity score measures are dot product and cosine similarity. |
| **Large Language Model (LLM)** | A model which has been trained to numerically represent the patterns in text. A generative LLM will continue a sequence when given a sequence.  For example, given a sequence of the text "hello, world!", a genertive LLM may produce "we're going to build a RAG pipeline today!".  This generation will be highly dependant on the training data and prompt. |
| **LLM context window** | The number of tokens a LLM can accept as input. For example, as of March 2024, GPT-4 has a default context window of 32k tokens  (about 96 pages of text) but can go up to 128k if needed. A recent open-source LLM from Google, Gemma (March 2024) has a context  window of 8,192 tokens (about 24 pages of text). A higher context window means an LLM can accept more relevant information  to assist with a query. For example, in a RAG pipeline, if a model has a larger context window, it can accept more reference items  from the retrieval system to aid with its generation. |
| **Prompt** | A common term for describing the input to a generative LLM. The idea of "prompt engineering" is to structure a text-based  (or potentially image-based as well) input to a generative LLM in a specific way so that the generated output is ideal. This technique is  possible because of a LLMs capacity for in-context learning, as in, it is able to use its representation of language to breakdown  the prompt and recognize what a suitable output may be (note: the output of LLMs is probable, so terms like "may output" are used). | 
