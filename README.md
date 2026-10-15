# Hotel Review Topic & Sentiment Analysis

NLP pipeline for analysing hotel reviews using BERTopic and sentence embeddings — built for the *Large Language Models for Marketing* course at Erasmus University Rotterdam.

## What It Does
1. **Topic Modelling** — Applies a pre-trained BERTopic model to hotel review sentences to assign each sentence a topic (e.g. room quality, staff, food, location)
2. **Sentiment Integration** — Combines topic assignments with sentence-level sentiment scores
3. **Customer Satisfaction Prediction** — Logistic regression using topic presence and topic-level sentiment to predict whether a reviewer is "happy"
4. **Brand Benchmarking** — Compares Hilton, Marriott, Sheraton, Hyatt, Holiday Inn, and Novotel across topics via:
   - Sentiment heatmaps (raw + topic-demeaned)
   - Spider / radar charts
   - PCA-based perceptual map

## Tech Stack
- Python 3
- `bertopic` · `sentence-transformers`
- `scikit-learn` · `statsmodels`
- `pandas` · `matplotlib` · `seaborn`

## Key Outputs
| Output | Description |
|---|---|
| Logit model summary | Topic presence + sentiment → customer happiness |
| `brand_topic_sentiment_heatmap.png` | Average sentiment per brand × topic |
| `brand_topic_sentiment_demeaned_heatmap.png` | Topic-demeaned brand sentiment |
| `brand_perceptual_map.png` | PCA map positioning brands by topic sentiment profile |
| `brand_topic_sentiment_spider.png` | Radar chart across brands and topics |

## Data
Hotel reviews dataset with `User_ID`, `Description` (review text), and `Is_Response` (happy / not happy). Sentence embeddings and BERTopic model weights are pre-computed and loaded from pickle files.
