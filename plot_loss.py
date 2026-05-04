import pandas as pd
import matplotlib.pyplot as plt

# 👉 UPDATE THIS PATH (your correct one)
csv_path = r"models\detection\yolo_train\results.csv"

# Load data
df = pd.read_csv(csv_path)

# Confirm rows
print("Total epochs logged:", len(df))

# Extract values
epochs = range(1, len(df) + 1)
train_loss = df["train/box_loss"]
val_loss = df["val/box_loss"]

# Professional style
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_loss, label="Training Loss", color="#0F0E47", linewidth=2.5)
plt.plot(epochs, val_loss, label="Validation Loss", color="#8686AC", linestyle="--", linewidth=2.5)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss Over Epochs", fontsize=14, fontweight="bold")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.fill_between(epochs, train_loss, val_loss, alpha=0.1)
# Clean look
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

# Save
plt.savefig("epoch_vs_loss_final.png", dpi=300, bbox_inches="tight")

print("Graph saved as epoch_vs_loss_final.png")

plt.show()