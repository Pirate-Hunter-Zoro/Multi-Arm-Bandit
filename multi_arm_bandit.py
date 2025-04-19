from matplotlib import pyplot as plt
import numpy as np
from BanditSimulator.bandit_sim import Bandit_Sim

bs = Bandit_Sim(n_arms=5, payout_std=0.1, seed=42)
# Let's analyze some payouts
payouts = [bs.pull_arm(i) for i in range(bs.n_arms)]
#         ----------
#         `num_samples` : int
#             the number of samples to plot
#         """
# Create a joint histogram of the payouts
num_samples = 100
joint_hist, xedges, yedges = np.histogram2d(payouts, payouts, bins=num_samples)
xedges = np.arange(len(bs.arm_means))
yedges = np.arange(len(bs.arm_means))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(joint_hist, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.title('Joint Histogram of Payouts')
plt.xlabel('Arm')
plt.ylabel('Payout')
plt.xticks(xedges)
plt.yticks(yedges)

plt.subplot(1, 2, 2)
plt.imshow(joint_hist.T, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.title('Joint Histogram of Payouts (Transposed)')
plt.xlabel('Payout')
plt.ylabel('Arm')
plt.xticks(xedges)
plt.yticks(yedges)
plt.tight_layout()
plt.savefig('Results/joint_bandit_histogram.png')