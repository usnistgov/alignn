"""Module to analyze FF training."""
from jarvis.db.jsonutils import loadjson
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

out_dir = "/wrk/knc6/ALINN_FC/FD_elements_mult/temp"
out_dir = "/wrk/knc6/ALINN_FC/FD_elements/temp"
out_dir = "/wrk/knc6/ALINN_FC/FD_mult/temp"
out_dir = "/wrk/knc6/ALINN_FC/FD/temp"
out_dir = "/wrk/knc6/ALINN_FC/FitSNAP/Ta_snap/Out2"
out_dir = "/wrk/knc6/ALINN_FC/FitSNAP/Ta_snap/Out3"
# out_dir = "/wrk/knc6/ALINN_FC/FitSNAP/Ta_snap/OutCu002"
out_dir = "/wrk/knc6/ALINN_FC/FitSNAP/Ta_snap/OutCu003"
out_dir = "/wrk/knc6/ALINN_FC/FitSNAP/Ta_snap/OutCu004"
# out_dir = "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113/out"
out_dir = "/wrk/knc6/ALINN_FC/ALIGNNFF_DB/temp"
out_dir = "/wrk/knc6/ALINN_FC/FD_mult/temp_new"
out_dir = "/wrk/knc6/ALINN_FC/FitSNAP/Ta_snap/OutMlearnAll"
# Plot training hostory
json_path = os.path.join(out_dir, "history_val.json")
v = loadjson(json_path)
ens = []
fs = []
for i in v:
    ens.append(i[0])
    fs.append(i[2])
the_grid = GridSpec(1, 2)
plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(12, 5))
plt.subplot(the_grid[0])
plt.title("(a) Energy")
plt.plot(ens)
plt.xlabel("Epochs")
plt.ylabel("eV")
plt.subplot(the_grid[1])
plt.title("(b) Forces")
plt.plot(fs)
plt.xlabel("Epochs")
plt.ylabel("eV/A")
plt.tight_layout()
plt.savefig("history.png")
plt.close()

# Plot val comparison
the_grid = GridSpec(1, 2)
json_path = os.path.join(out_dir, "Val_results.json")
test = loadjson(json_path)
plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(12, 5))
plt.subplot(the_grid[0])
xx = []
yy = []
factor = 1
for i in test:
    for j, k in zip(i["target_out"], i["pred_out"]):
        xx.append(j)
        yy.append(k)
xx = np.array(xx) * factor
yy = np.array(yy) * factor

x_bar = np.mean(xx)
baseline_mae = mean_absolute_error(
    np.array(xx),
    np.array([x_bar for i in range(len(xx))]),
)
print("Val")
print("Baseline MAE: eV", baseline_mae)
print("MAE eV", mean_absolute_error(xx, yy))

plt.plot(xx, yy, ".")
plt.ylabel("ALIGNN Energy (eV)")
plt.xlabel("DFT Energy (eV)")


plt.subplot(the_grid[1])
xx = []
yy = []
for i in test:
    for j, k in zip(i["target_grad"], i["pred_grad"]):
        for m, n in zip(j, k):
            xx.append(m)
            yy.append(n)
xx = np.array(xx) * factor
yy = np.array(yy) * factor

x_bar = np.mean(xx)
baseline_mae = mean_absolute_error(
    np.array(xx),
    np.array([x_bar for i in range(len(xx))]),
)
print("Test")
print("Baseline MAE: eV/A", baseline_mae)
print("MAE eV/A", mean_absolute_error(xx, yy))
plt.scatter(xx, yy, c="blueviolet", s=10, alpha=0.5)

plt.scatter(xx, yy, c="blueviolet", s=10, alpha=0.5)
plt.ylabel("ALIGNN Force (eV/A)")
plt.xlabel("DFT Force (eV/A)")
plt.tight_layout()
plt.savefig("val.png")
plt.close()


# Plot train comparison
the_grid = GridSpec(1, 2)
json_path = os.path.join(out_dir, "Train_results.json")
test = loadjson(json_path)
plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(12, 5))
plt.subplot(the_grid[0])
xx = []
yy = []
factor = 1
for i in test:
    for j, k in zip(i["target_out"], i["pred_out"]):
        xx.append(j)
        yy.append(k)
xx = np.array(xx) * factor
yy = np.array(yy) * factor

x_bar = np.mean(xx)
baseline_mae = mean_absolute_error(
    np.array(xx),
    np.array([x_bar for i in range(len(xx))]),
)
print("Train")
print("Baseline MAE: eV", baseline_mae)
print("MAE eV", mean_absolute_error(xx, yy))

plt.plot(xx, yy, ".")
plt.ylabel("ALIGNN Energy (eV)")
plt.xlabel("DFT Energy (eV)")


plt.subplot(the_grid[1])
xx = []
yy = []
for i in test:
    for j, k in zip(i["target_grad"], i["pred_grad"]):
        for m, n in zip(j, k):
            xx.append(m)
            yy.append(n)
xx = np.array(xx) * factor
yy = np.array(yy) * factor

x_bar = np.mean(xx)
baseline_mae = mean_absolute_error(
    np.array(xx),
    np.array([x_bar for i in range(len(xx))]),
)
print("Baseline MAE: eV/A", baseline_mae)
print("MAE eV/A", mean_absolute_error(xx, yy))
plt.scatter(xx, yy, c="blueviolet", s=10, alpha=0.5)

plt.scatter(xx, yy, c="blueviolet", s=10, alpha=0.5)
plt.ylabel("ALIGNN Force (eV/A)")
plt.xlabel("DFT Force (eV/A)")
plt.tight_layout()
plt.savefig("train.png")
plt.close()
