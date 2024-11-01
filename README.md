<h1 align="center"> NLP with Hugging Face Transformers</h1>
<p align="center"> Natural Language Processing (NLP)

<p align="center">
  <a

![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)

<p align="center">
  <a

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naereen/badges)
[![Visual Studio](https://badgen.net/badge/icon/visualstudio?icon=visualstudio&label)](https://visualstudio.microsoft.com)

</p>

<p align="center"> 🧠 What is NLP?</p>
Natural Language Processing (NLP) is one of the main fields in artificial intelligence (AI) that enables machines to understand and respond to human language naturally. With NLP, we can build applications that interact effectively with users, assist in text analysis, and support various automated functions in business, research, and everyday life.

## 📄 The goal of NLP (Natural Language Processing)
    NLP aims to bridge communication between humans and computers. By using NLP techniques, computers can:

    1. Analyze and understand the meaning of text.

    2. Identify important entities such as names, locations, and organizations.

    3. Measure emotions and sentiments in the text.

    4. Generate text that sounds natural in many languages."

## 🚀 Technologies & Methods in NLP
    Several key technologies and approaches in NLP include:

    1. Machine Learning
       Traditional machine learning models such as Naive Bayes, SVM, and Random Forest are often used in classic NLP applications.

    2. Deep Learning
       Neural network models like LSTM and CNN help computers understand context and sentence structure.

    3. Transformers
       Transformer models have revolutionized NLP by introducing models like BERT, GPT, and T5, which can handle long and complex texts and support NLP tasks more effectively.

    4. Text Preprocessing
       NLP requires preprocessing stages such as tokenization, stemming, lemmatization, and stop-word removal to ensure data is ready for model use.

## 🤖 Applications of NLP in Real Life
    NLP has various highly beneficial applications, such as:

    1. Chatbots & Virtual Assistants: Assisting users with instant information and 24/7 customer service.

    2. Sentiment Analysis: Understanding public opinion from product reviews, surveys, or social media.

    3. Language Translation: Enabling real-time translation of text or speech.

    4. Information Extraction: Extracting important information from lengthy documents, such as financial reports or articles.

    5. Text Summarization: Generating summaries from long texts, useful for news or reports.

## 🛠️ Tools and Libraries Used in NLP
   <p align="center">
    <a
    <p>
      <img src="https://img.shields.io/badge/-NLTK-777BB4?style=flat&logo=python&logoColor=white" />
      <img src="https://img.shields.io/badge/-spaCy-09A3D5?style=flat&logo=python&logoColor=white" />
      <img src="https://img.shields.io/badge/-Transformers-FF6F00?style=flat&logo=python&logoColor=white" />
      <img src="https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white" />
      <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
      <img src="https://img.shields.io/badge/-Hugging%20Face-F7B8D3?style=flat&logo=hugging-face&logoColor=black" />
      </p>

## 📈 NLP in the Modern Era: Popular Models
    Here are some of the most influential models:

    1. BERT (Bidirectional Encoder Representations from Transformers)
       Developed by Google, BERT introduced a transformer architecture that allows for bidirectional context understanding, significantly improving performance on various NLP tasks such        as question answering and sentiment analysis.
   
    2. GPT (Generative Pre-trained Transformer)
       Created by OpenAI, GPT is a generative model known for its ability to produce coherent and contextually relevant text. The model's architecture enables it to handle tasks ranging        from text completion to creative writing.
   
    3. T5 (Text-to-Text Transfer Transformer)
       Also developed by Google, T5 treats all NLP tasks as a text-to-text problem, allowing for a unified approach to diverse tasks like translation, summarization, and classification.
   
    4. RoBERTa (Robustly Optimized BERT Approach)
       An extension of BERT, RoBERTa enhances the training process by using more data and removing the next sentence prediction objective, leading to improved performance on benchmarks.
   
    5. XLNet
       A generalized autoregressive pretraining model, XLNet combines the strengths of both autoregressive and autoencoding models, allowing it to capture bidirectional contexts and            perform e3xceptionally well on various NLP tasks.
   
    6. ALBERT (A Lite BERT)
       Aimed at reducing the model size while maintaining performance, ALBERT introduces parameter-sharing techniques and factorized embeddings, making it more efficient than its       
       predecessor, BERT.
    
    7. ERNIE (Enhanced Representation through kNowledge Integration)
       Developed by Baidu, ERNIE focuses on integrating knowledge graphs into the training process, improving the model's understanding of language and context.
    
    8. DistilBERT
       A smaller, faster, and lighter version of BERT, DistilBERT retains 97% of BERT’s language understanding while being 60% faster, making it suitable for applications where speed is        crucial.
## 📊 NLP Statistics
    With NLP, we can collect and analyze data from various sources in large quantities. Here are some examples of how NLP statistics are used:
    1. Sentiment Analysis to measure customer opinions or satisfaction.
    2. Word and Topic Frequency to understand current trending topics.
    3. Text Classification to group documents or automate message categorization.

## 💻 How to Start Using NLP in This Project
  Library Installation: Make sure to install the following libraries:
    ```bash
     pip install nltk spacy transformers torch</p>

   <div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">

</div>

---
<h2 align="center"> The Result Analyze </h2> 

  1. Zero-shot classification is an NLP technique that enables a model to classify text into categories it has not been trained on. The text "I'm feeling very grateful for all the       
     support from my friends." is identified with the label gratitude 93.33%, indicating that the sentence is most relevant to gratitude. 

  2. Text generation is the process of creating new text. The sentence "The greatest lesson I learned this year is that" the model will continue "you have a choice. You say, 'Okay,   
     that's my choice, or I'll get to go back.'"

  3. Fill-mask enables a model to guess missing words. In the sentence "The new policy will affect the [MASK] sector significantly," the model predicts the word "private" with a score 
     of 38.4%.

  4. Named Entity Recognition (NER) identifies entities in text. In the sentence ""Barack Obama was born in Hawaii and was the 44th president of the United States." The model detects   
     "Barack Obama" as a person and "Hawaii", and "United States" as locations, with confidence scores above 0.99 indicates very high.

  5. Question-answering (QA) enables a model to answer questions based on context. For a sentence about Bali, when asked "What are some popular tourist spots in Bali?" the model answers 
     "Uluwatu Temple, Tanah Lot."

  6. Sentiment analysis classifies emotions in text. The sentence "The movie had stunning visuals but the plot was quite predictable" is labeled NEGATIVE with high confidence (99.93%) 
     because, despite the praise, the complaint about the plot is dominant.

  7. Summarization condenses lengthy text into shorter, meaningful content. A model can summarize a lengthy report to highlight key information.

  8. Translation involves translating text from one language to another. The original sentence "Indonesia adalah negara kepulauan terbesar di dunia, terdiri dari lebih dari 17.000     
     pulau. Negara ini terkenal dengan keanekaragaman budaya dan alamnya." is translated as "Indonesia is the world's largest island nation, made up of over 17,000 islands, known for   
     its cultural and natural diversity."

Of all these techniques, Zero-shot classification and Sentiment analysis can be considered more interesting because they not only classify but also provide deeper insights into the meaning and emotions behind the text, which is invaluable in data analysis and decision-making.








