# Sentiment-Financial-News-Analysis
Financial News Sentiment Analysis using NLP. Achieved 0.9 F1 with RoBERTa. Explore models: SVM, BERT, XLNet, RoBERTa.
![Sentiment Analysis](https://github.com/joshichakresh/Sentiment-Financial-News-Analysis/blob/main/Screenshot%20(289)3.png)
![F1-Score](https://github.com/joshichakresh/Sentiment-Financial-News-Analysis/blob/main/Screenshot%20(291)2.png)
![Testing](https://github.com/joshichakresh/Sentiment-Financial-News-Analysis/blob/main/Screenshot%202023-07-22%20133555.png)


## Overview
This project focuses on sentiment analysis of financial news headlines using advanced NLP models. The goal is to determine the most effective model for this task, with a primary focus on maximizing the F1 score.

## Model Performance
After training and evaluating various models, RoBERTa stands out as the top performer, achieving an impressive F1 score of 0.91. Other models explored include SVM, BERT and XLNet.

## Dataset
The dataset used for this project comprises financial news headlines annotated with sentiment classes (positive, negative, neutral). Rigorous preprocessing was employed to ensure the data's quality and suitability for training and evaluation. It's worth noting that the dataset was sourced from a licensed dataset on Kaggle, ensuring adherence to copyright and licensing terms.

## Repository Structure
- `SVM/`: Support Vector Machines model code and files.
- `BERT/`: BERT model code and files.
- `XLNet/`: XLNet model code and files.
- `RoBERTa/`: RoBERTa model code and files.
- `data/`: Raw dataset for training and evaluation

## Usage
Please refer to the respective model directories for detailed setup instructions and usage guidelines. Choose RoBERTa for the best results.

## Future Improvements
FinRoBERTa: Further exploration and fine-tuning of the "FinRoBERTa" model, specifically designed for financial sentiment analysis, hold promise for improved performance and better capturing of financial domain nuances. This model's potential lies in enhancing the ability to extract meaningful insights from financial news headlines, contributing to even more accurate sentiment analysis.

## Conclusion
This project demonstrates cutting-edge sentiment analysis on financial news headlines. RoBERTa excels among the models. Contributions are welcome!
