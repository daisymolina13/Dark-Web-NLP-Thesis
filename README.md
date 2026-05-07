# Black Market Evolution Using NLP
**Pace University Honors Thesis**
Author: Daisy Molina | Faculty Advisor: Carmine Guida

## Overview
This project applies Natural Language Processing (NLP) techniques to analyze communication patterns within dark web marketplaces ecosystems. Using datasets of 194,433 listings across 8 platforms, this study identifies behavioral clusters, infers personality traits, and tracks linguistic drift over time.

## Research Questions
- Can NLP techniques reveal distinct seller behavior profiles in dark web marketplaces?
- Do personality traits inferred from text vary across platforms?
- How does seller communication evolve across different marketplaces?

# Dataset
- **Source:** Agora Darknet Market Dataset (Kaggle)
- **Size:** 194,433 listings
- **Platforms covered:** Agora, Silk Road 2, Nucleus, Evolution, Outlaw Market, Abraxas, The Marketplace, 1776
- Note: Dataset is not included due to its file size, link: ([https://www.kaggle.com/datasets/mhwong2007/drug-listing-dataset])

## Methodology
1. **Preprocessing** - tokenization, lemmatization, stop-word removal using SpaCy
2. **Feature Extraction** - TF-IDF vectorization (top 1000 features)
3. **Clustering** - K-Means (k=5) to identify seller behavioral groups 
4. **Personality Profiling** - Big Five model applied via lexicon-based scoring
5. **Longitudinal Analysis** - personality drift tracked across 8 platforms

## Key Findings
- 5 distinct seller clusters identified based on linguistic patterns
- Stealth/shipping vendors scored highest in conscientiousness (3.64)
- Hard drug vendors scored highest in agreeableness (1.40)
- Agora showed highest agreeableness across platforms (1.69)
- Neuroticism remained consistently low across all platforms

## Results
![Personality by Cluster]()
![Personality Drift]()
![Listings per Source]()

## Tools & Libraries
- Python 3
- spaCy
- scikit-learn
- pandas
- matplotlib
- seaborn

## Ethical Considerations
This study uses only publicly available archived data. No live dark web marketplaces were accessed. No personally identifiable information was collected. All findings were found and reported in mass. 
