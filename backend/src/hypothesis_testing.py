import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, shapiro, probplot

# Your average nDCG@k values (replace with actual scores)
ndcg_tfidf = [0.689, 0.742, 0.755, 0.761, 0.767, 0.767, 0.765, 0.758, 0.755, 0.753]
ndcg_hybrid = [0.702, 0.745, 0.764, 0.763, 0.768, 0.767, 0.764, 0.763, 0.760, 0.755]

# Step 1: Compute paired differences
differences = np.array(ndcg_hybrid) - np.array(ndcg_tfidf)

# Step 2: Save plot instead of showing
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(differences, kde=True, bins=7, color='skyblue')
plt.axvline(0, color='red', linestyle='--')
plt.title("Histogram of Differences (Hybrid - TF-IDF)")
plt.xlabel("Difference in nDCG@k")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
probplot(differences, dist="norm", plot=plt)
plt.title("Q-Q Plot of Differences")

plt.tight_layout()
plt.savefig("full_output/ndcg_diff_analysis.png", dpi=300)      # Change the path here.
plt.close()  # Close the plot to avoid displaying in interactive environments

print("Plot saved as 'ndcg_diff_analysis.png' with 300 DPI.")


# Step 3: Normality test (Shapiro-Wilk)
shapiro_stat, shapiro_p = shapiro(differences)
print("Shapiro-Wilk Normality Test:")
print(f"  W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")


# Step 4: Choose appropriate test
if shapiro_p > 0.05:
    print("Differences appear normally distributed (p > 0.05)")
    # Use paired t-test (one-tailed)
    t_stat, p_val_two_tailed = ttest_rel(ndcg_hybrid, ndcg_tfidf)
    p_val_one_tailed = p_val_two_tailed / 2
    print("\n Paired t-test:")
    print(f"  t-statistic = {t_stat:.4f}, one-tailed p-value = {p_val_one_tailed:.4f}")
    if p_val_one_tailed < 0.05:
        print("Result: HYBRID significantly outperforms TF-IDF.")
    else:
        print("Result: No significant difference.")
else:
    print("Differences not normally distributed (p ≤ 0.05)")
    # Use Wilcoxon signed-rank test (non-parametric, one-tailed)
    w_stat, w_p = wilcoxon(differences, alternative='greater')
    print("\n Wilcoxon signed-rank test:")
    print(f"  W-statistic = {w_stat:.4f}, one-tailed p-value = {w_p:.4f}")
    if w_p < 0.05:
        print(" Result: HYBRID significantly outperforms TF-IDF.")
    else:
        print(" Result: No significant difference.")
