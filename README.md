# Fake News Detection

## Overview
This project aims to tackle the growing issue of fake news by developing a machine learning model that classifies news as either fake or real. By leveraging various feature extraction methods and machine learning algorithms, the system can accurately identify misleading or false information.

## Features
- **Data Collection**: The dataset used is a combination of real and fake news articles, primarily sourced from Kaggle.
- **Preprocessing**: The text data undergoes rigorous preprocessing, including lowercasing, removal of numbers, punctuation, stop words, and more.
- **Feature Extraction**: Techniques like Bag of Words (BOW), Term Frequency-Inverse Document Frequency (TF-IDF), and Word Embedding are used to convert text data into feature vectors.
- **Classification Algorithms**: Several machine learning algorithms are implemented, including:
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - Naive Bayes
- **Model Evaluation**: Models are evaluated based on accuracy, precision, recall, and F1-score using confusion matrices and other metrics.

## Results
- The project achieves high accuracy in detecting fake news, with the SVM classifier combined with TF-IDF feature extraction showing the best performance, reaching an accuracy of 94%.
- Other models like Random Forest, Decision Tree, and Naive Bayes also perform well, but SVM stands out in terms of both efficiency and accuracy.

## Future Scope
- **Deep Learning**: Incorporating deep learning techniques such as Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for improved accuracy.
- **Sentiment Analysis**: Adding sentiment analysis to enhance the model's ability to detect fake news.
- **Extended Features**: Including more sophisticated features like publication source, URL domain, and others to further refine the classification process.

## Installation
1. Clone the repository.
2. Install required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebooks or scripts to preprocess data, train models, and evaluate performance.

## Usage
- Place your news article text in the input box, and the model will predict whether the news is real or fake.
- The repository includes sample scripts and a notebook demonstrating the entire process from data preprocessing to model evaluation.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License. 
