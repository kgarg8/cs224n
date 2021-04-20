import matplotlib.pyplot as plt, numpy as np

train_loss = np.array([0.194226679968692, 0.11617550633654192, 0.1015650380636642, 0.09306087676290568, 0.08676497005047155, 0.081329650995503, 0.07734990064296629, 0.073567983111794, 0.07033381995732908, 0.06723299680189007])

dev_UAS = np.array([83.78, 85.73, 86.39, 87.49, 88.12, 88.30, 88.16, 88.39, 88.60, 88.55])

epochs = np.arange(1, 11)

plt.xlabel('Epochs')
plt.ylabel('Train loss')
plt.title('Train loss with epochs')

plt.plot(epochs, train_loss, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
for a,b in zip(epochs, train_loss): 
    plt.text(a, b, str(round(b, 3)))

plt.show()

plt.xlabel('Epochs')
plt.ylabel('Dev_UAS')
plt.title('Unlabeled Attachment Score on Dev Set with epochs')
plt.plot(epochs, dev_UAS, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
for a,b in zip(epochs, dev_UAS): 
    plt.text(a, b, str(b))
plt.show()