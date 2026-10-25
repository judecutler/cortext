import sys; sys.path.append("datasets/cortext-p300")
from cortext_data import load_epochs, to_mne
import matplotlib.pyplot as plt

X, y, info = load_epochs()
print(X.shape, "targets:", int(y.sum()), "non-targets:", int((y == 0).sum()))
print("distinct trial groups:", len(set(info["groups"])))

ep = to_mne(X, info, y)
d, t = ep.get_data(), ep.times
pz = ep.ch_names.index("Pz")
plt.plot(t, d[y == 1][:, pz].mean(0), label="target")
plt.plot(t, d[y == 0][:, pz].mean(0), label="non-target")
plt.axvline(0, color="gray", ls=":"); plt.legend(); plt.title("Pz"); plt.show()