import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv("test_scores/gemini_vs_memoirs.csv")
df = df.sort_values("custom_alignment")

plt.plot(df["custom_alignment"].values, label="Detailed Custom Model")
plt.plot(df["gemini_alignment"].values, label="gemini")
plt.legend()
plt.title("BERTScore Alignment with BLIP Caption")
plt.ylabel("BERT F1 Score")
plt.xlabel("Images (sorted by custom model score)")
plt.grid(True)
plt.show()

