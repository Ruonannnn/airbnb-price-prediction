# Optimizing Airbnb Price Prediction with Machine Learning

This project develops machine learning-based predictive models that integrates diverse listing attributes,such as room type, geographic location, and availability patterns, to estimate optimal prices and further to support Airbnb hosts in implementing dynamic, data-informed pricing strategies that enhance competitiveness and maximize revenue.

## Motivation

Many Airbnb hosts currently rely on their own intuition or casually compare prices with a few similar listings to determine the price for their property. This often leads to inconsistent pricing strategies, suboptimal revenue, and lost competitive advantage in the market.
By introducing a data-driven pricing recommendation feature, we aim to help hosts improve booking rates and support Airbnb in maximizing occupancy and overall profitability.

## Dataset

- **Source**: [Airbnb Open Data Dataset (Kaggle)](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata/data)
- 60,000+ listings after cleaning
- **Key Variables**: History Price, Room Type, Number of Bedrooms/Bathrooms, Location (latitude/longitude), Availability, construction year, and various amenities
- **Data Cleaning**: removed irrelevant or duplicate columns, handled missing values, standardized column names, converted prices and dates to proper formats, and encoded categorical variables.

### Exploratory Data Analysis (EDA)

#### Geographical Availability
Neighborhoods were categorized based on listing availability:

- **Low Availability** (e.g., Bay Terrace, Downtown Brooklyn): High demand → Higher pricing potential
- **Medium Availability** (e.g., Williamsburg, East Village): Balanced supply-demand → Use dynamic pricing
- **High Availability** (e.g., Bath Beach, Willowbrook): High vacancy → Require competitive pricing
<p align="center">
  <img src="distribution map.png" width="600">
</p>

#### Correlation Analysis
- A correlation heatmap revealed **weak pairwise correlations** between most features and the target `price`.
- Features like `number_of_reviews`, `minimum_nights`, `availability_365`, and `review_scores` had **low explanatory power**, implying the need for **nonlinear and interaction-aware models**.
