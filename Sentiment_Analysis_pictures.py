import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Before labeling
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'], rotation=0)
plt.title("Class Distribution After Auto-Labeling")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Get feature importance from XGBoost
importances = best_xgb.feature_importances_
features = vectorizer.get_feature_names_out()

# Sort top 20 important words
sorted_idx = np.argsort(importances)[-20:]
plt.figure(figsize=(8, 5))
plt.barh(range(20), importances[sorted_idx], tick_label=[features[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Word")
plt.title("Top 20 Important Words for Sentiment Prediction")
plt.show()
