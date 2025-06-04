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

- **Low Availability**: High demand → Higher pricing potential
- **Medium Availability**: Balanced supply-demand → Use dynamic pricing
- **High Availability**: High vacancy → Require competitive pricing

#### Correlation Analysis
- A correlation heatmap revealed **weak pairwise correlations** between most features and the target `price`.
- Features like `number_of_reviews`, `minimum_nights`, `availability_365`, and `review_scores` had **low explanatory power**, implying the need to consider for **nonlinear and interaction-based models**.

## Modeling Approach

We implemented and compared three models to predict continuous Airbnb listing prices:

| Model              | Description |
|-------------------|-------------|
| **Linear Regression** | Served as a baseline model. It captured basic relationships but failed to handle complex nonlinear interactions. Feature coefficients were small and model performance was poor. |
| **LASSO Regression (with interaction terms)** | Helped reduce noise and select the most relevant features. It performed slightly better than linear regression but still struggled due to weak correlations in the data. |
| **Random Forest** | Delivered the best results by capturing nonlinear relationships and complex interactions. It uses recursive feature splits rather than relying on explicit coefficients, making it robust against weak individual feature correlations. |


## Evaluation

We used **K-Fold Cross-Validation** to evaluate model performance and reduce variance from random data splits.

- **Primary metric**: Out-of-sample R² (R-squared on test folds)
- **Observation**: Despite Random Forest being the best among tested models, overall R² remained low due to:

  - Weak pairwise correlations between features and target
  - High heterogeneity in listing types and guest preferences
  - Missing contextual features such as seasonality or proximity to events

## Limitations & Future Improvements

### Limitations:
- **Missing Influential Features**: No data on seasonality, special events, or proximity to attractions.
- **Weak Feature Correlations**: Most retained variables had low direct impact on price.
- **Reduced Data Size**: Cleaning removed outliers and illogical entries but reduced diversity in training data.

### Next Steps:
- Enhance dataset by adding:
  - Temporal variables (e.g., holidays, peak travel seasons)
  - Geographic context (e.g., tourist spots, subway distance)
  - Amenity details (e.g., parking, Wi-Fi, pool)
- Explore ensemble models or time-series-aware approaches
- Evaluate success through business metrics like:
  - Change in **booking rate**
  - Increase in **host revenue**
  - Boost in **Airbnb platform commission**
