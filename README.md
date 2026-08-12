# Soccer Player Valuation Analysis

> **Scope** · Self-study data science project — published as-is.

> [!NOTE]
> **Built 2025. The ecosystem has moved since.**
> Dependencies here are unpinned, so a clean `pip install` today resolves to
> versions that did not exist when this was written and pandas 3.0, numpy 2.5, pytest 9 and black 26 have all landed since. Expect install or
> runtime breakage on a fresh environment. What is on offer is the engineering
> approach and the decisions behind it, not a guaranteed-green build.
> Happy to bring it current if that would be useful — just ask.

## Overview
This project analyzes and predicts soccer player market values using performance statistics, clustering techniques, and machine learning. The system processes player appearances, club data, and valuation history to provide insights into player segmentation and market value prediction.

## Key Features
- **Data Integration**: Combines player stats, club info, and valuation history
- **Player Segmentation**: K-means clustering identifies 5 distinct player profiles
- **Market Value Prediction**: XGBoost model predicts values with R²=0.87
- **Performance Analysis**: Visualizations of key metrics across clusters
- **Production-Ready**: Includes custom prediction functionality

## Technical Implementation
### Data Processing
```python
# Merged multiple datasets (players, appearances, games)
full_players_info = players.merge(appearences, on='player_id')
full_players_info = full_players_info.merge(games[['game_id','season']], on='game_id')

# Calculated age from birth dates
full_players_info['age'] = (today - full_players_info['date_of_birth']).dt.days / 365.25
```

### Clustering Analysis
```mermaid
graph TD
    A[Raw Data] --> B[Feature Selection]
    B --> C[Data Scaling]
    C --> D[Elbow Method]
    D --> E[K-Means Clustering]
    E --> F[Cluster Analysis]
```

### Machine Learning
- XGBoost with GPU acceleration
- GridSearchCV for hyperparameter tuning
- Feature importance analysis
- Custom prediction function
