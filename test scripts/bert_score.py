import os
import pandas as pd
from bert_score import score
import matplotlib.pyplot as plt

# Folders
custom_dir = "custom_outputs"
gpt4o_dir = "gpt4o_outputs"

# output date
results = []

# iteration through caption files
for filename in os.listdir(custom_dir):
    if not filename.endswith("_caption.txt"):
        continue

    base = filename.replace("_caption.txt", "")
    caption_path = os.path.join(custom_dir, filename)
    custom_story_path = os.path.join(custom_dir, f"{base}_custom_story.txt")
    gpt4o_story_path = os.path.join(gpt4o_dir, f"{base}_gpt4o_story.txt")

    if not os.path.exists(custom_story_path) or not os.path.exists(gpt4o_story_path):
        continue

    # read texts
    with open(caption_path, "r", encoding="utf-8") as f:
        raw_caption = f.read().strip()
    with open(custom_story_path, "r", encoding="utf-8") as f:
        text_custom = f.read().strip()
    with open(gpt4o_story_path, "r", encoding="utf-8") as f:
        text_gpt4o = f.read().strip()

    # Use only first sentence of BLIP caption as ground truth
    caption = raw_caption.split('.')[0].strip() + '.'

    # bert score part
    try:
        candidates = [text_custom, text_gpt4o]
        references = [caption, caption]

        P, R, F1 = score(candidates, references, lang="en", verbose=False, rescale_with_baseline=False)

        results.append({
            "image": base,
            "caption": caption,
            "custom_precision": round(P[0].item(), 4),
            "custom_recall": round(R[0].item(), 4),
            "custom_f1": round(F1[0].item(), 4),
            "gpt4o_precision": round(P[1].item(), 4),
            "gpt4o_recall": round(R[1].item(), 4),
            "gpt4o_f1": round(F1[1].item(), 4)
        })

    except Exception as e:
        print(f"⚠️ Skipping {base} due to error: {e}")
        continue

# saving results
df = pd.DataFrame(results)
df.to_csv("bert_alignment_scores.csv", index=False)
print("✅ Saved BERTScore results to 'bert_alignment_scores.csv'")

# Visualize the data
df_sorted = df.sort_values("custom_f1").reset_index(drop=True)
x = range(len(df_sorted))

plt.figure(figsize=(12, 6))
plt.plot(x, df_sorted["custom_f1"], label="Custom Model", marker='o')
plt.plot(x, df_sorted["gpt4o_f1"], label="GPT-4o", marker='x')
plt.xlabel("Images (sorted by custom model score)")
plt.ylabel("BERT F1 Score")
plt.title("BERTScore Alignment with BLIP Caption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bert_alignment_plot.png")
plt.show()
