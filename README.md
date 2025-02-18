# **Video Transcript Extraction and Summarization using Transfer Learning**

## **Overview**
This project focuses on **automated video transcript extraction and summarization** using **state-of-the-art NLP models and Transfer Learning techniques**. It aims to provide concise, high-quality summaries of video content, making information more accessible and reducing the need for users to watch entire videos.

## **Key Features**
✅ **Automatic Speech Recognition (ASR)** – Converts video audio into text using **Google Speech-to-Text API**  
✅ **Pre-trained Transformer-based Summarization** – Fine-tuned **BART-Large-CNN** model for **abstractive text summarization**  
✅ **Efficient Data Processing** – Uses **Pandas** and **NLTK** for text cleaning and preprocessing  
✅ **Multi-metric Evaluation** – Summarization quality evaluated with **ROUGE Score** and **BERT Score**  
✅ **Scalability & Extensibility** – Can be expanded to support **multi-lingual transcript summarization**  

## **Technologies Used**
- **Programming Language:** Python  
- **Machine Learning Frameworks:** HuggingFace Transformers, TensorFlow, PyTorch  
- **Natural Language Processing (NLP):** BART-Large-CNN, T5, ROUGE, BERT Score  
- **Speech-to-Text Processing:** Google Speech-to-Text API  
- **Data Handling:** Pandas, NumPy, NLTK  
- **Visualization:** Matplotlib  

## **Project Workflow**
1. **Video Processing:** Extracts **audio** from the video file.  
2. **Automatic Speech Recognition (ASR):** Converts audio into **text transcripts**.  
3. **Text Preprocessing:** Cleans transcripts using **lowercasing, punctuation removal, and stopword filtering**.  
4. **Summarization Model:** Applies **fine-tuned BART-Large-CNN** to generate **abstractive summaries**.  
5. **Evaluation Metrics:** Uses **ROUGE Score** and **BERT Score** to assess summarization quality.  

## **Dataset**
- **How2 Dataset** – Contains **8,000 YouTube videos** with corresponding **human-written summaries**  
- **Fine-Tuning:** The **BART-Large-CNN** model was **fine-tuned** using a subset of this dataset to improve summarization quality for video transcripts.  

## **Performance Metrics**
| **Model** | **ROUGE-1** | **ROUGE-2** | **BERT Score (F1)** |
|-----------|------------|------------|---------------------|
| T5-Small | 0.226 | 0.05 | 0.9505 |
| T5-Base | 0.233 | 0.048 | 0.9514 |
| **BART-Large-CNN (Fine-Tuned)** | **0.43** | **0.19** | **0.884** |

- **Fine-tuned BART-Large-CNN model** achieved **higher ROUGE and BERT Scores** compared to other models, making it the best choice for abstractive summarization.

## **Future Enhancements**
🔹 **Support for multiple languages** to expand beyond English transcripts  
🔹 **Integration with real-time streaming services** for live summarization  
🔹 **Improved summarization models** by experimenting with **GPT-4, Pegasus, and T5-XXL**  
🔹 **User-friendly API** for integrating summarization features into web applications  

## **Contributors**
- **Varun Mehta** – **ASR pipeline, fine-tuning BART, evaluation metrics**  
- **Tushar Deshpande** – **Data preprocessing, summarization model training**  
- **Ravikumar Pandey** – **Evaluation and optimization of BERT Score metrics**  
- **Tanishq Kandoi** – **Dataset analysis, data augmentation, and testing**  
- **Sindhu Nair** – **Speech-to-text processing, dataset preparation**  

---
🚀 **This project provides a robust, AI-powered solution for video summarization, helping users access information faster and more efficiently.**  
