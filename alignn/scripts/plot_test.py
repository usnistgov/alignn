import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("agg")
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("prediction_results_test_set.csv")
target = df.target.values
pred = df.prediction.values
df["abs_diff"] = (df["target"] - df["prediction"]).abs()
plt.plot(target, target,c='g')
plt.scatter(target, pred, c="blue",alpha=0.5,s=1)
title = (
    "R2,MAE:"
    + str(round(r2_score(target, pred), 3))
    + ","
    + str(round(mean_absolute_error(target, pred), 2))
)
plt.title(title)
plt.xlabel("Actual values")
plt.ylabel("DL values")
plt.tight_layout()
dat_lim = [0, 20]
plt.xlim(dat_lim)
plt.ylim(dat_lim)
plt.savefig("testset.png")
plt.close()
print("Worst 5 materials")
worst_ids = np.array(
    (df.sort_values("abs_diff")[::-1][0:5].values[:, 0]), dtype="int"
).tolist()
print(worst_ids)
print(
    "r2_score,mean_absolute_error",
    r2_score(target, pred),
    mean_absolute_error(target, pred),
)
