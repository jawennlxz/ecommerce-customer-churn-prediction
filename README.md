# E-commerce Customer Churn Prediction

Prepares a dataset of 3,405 online retail customers, explores which factors relate to churn, and builds a CART decision tree to predict which customers are likely to leave. The decision rules are then turned into retention recommendations.

Module: ANL252 Python for Data Analytics, Singapore University of Social Sciences (2026)

## Approach

1. **Data preparation**
   - Replaced the placeholder value `Missing?` in `Customer_Segment` with a true missing value, then filled the gaps with the mode.
   - Removed outliers in quantity, price, age and delivery time using the 1.5 × IQR rule.
   - Encoded yes/no fields as 1/0 and the other categories as numeric codes.
   - Engineered new features: `Total_Spend`, `Avg_Spend_Per_Item` and `Is_Frequent_Buyer`.
2. **Exploratory analysis**
   - Plotted single-variable ROC curves to compare each variable's predictive power.
   - Built a correlation heatmap.
   - Charted churn by customer segment and ran chi-square tests.
3. **Modelling:** trained CART decision trees (Gini impurity) at depths 3, 5, 7 and 10 on a 70/30 train-test split.
4. **Interpretation:** extracted the depth-7 tree's rules to explain what drives churn.

## Results

| Tree depth | Train accuracy | Train ROC-AUC | Test accuracy | Test ROC-AUC |
|---|---|---|---|---|
| 3 | 0.651 | 0.559 | 0.629 | 0.502 |
| 5 | 0.670 | 0.617 | 0.609 | 0.510 |
| **7** | **0.702** | **0.691** | **0.613** | **0.530** |
| 10 | 0.758 | 0.811 | 0.584 | 0.527 |

Deeper trees fit the training data better but did worse on new data, a sign of overfitting. Depth 7 had the best test ROC-AUC, so it was chosen as the final model.

**Key findings**

- No single variable predicts churn well; every single-variable AUC is close to 0.5. Churn comes from a combination of factors.
- Chi-square tests found no significant differences in churn between customer segments, so churn affects all segments.
- Customers with very low spend and longer delivery times are more likely to churn.
- Recommendations: minimum-spend incentives, discounts paired with faster delivery guarantees, and personalised offers for younger, low-spend customers.

## How to run

```bash
pip install pandas matplotlib scikit-learn scipy jupyter
jupyter notebook customer_churn_analysis.ipynb
```

The notebook expects `ECA_churn.csv` in the same folder. The dataset was provided for coursework and is not included here.

## Skills demonstrated

- Data cleaning, outlier treatment and feature engineering with pandas
- Exploratory analysis with ROC curves, correlation and chi-square tests
- Decision tree modelling and tuning with scikit-learn
- Turning model rules into business recommendations

## Author

Jawenn Lim
