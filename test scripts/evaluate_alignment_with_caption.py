import os
import pandas as pd
from bert_score import score as bert_score

# Folders
custom_dir = "detailed_custom_outputs"
gpt4o_dir = "gpt4o_outputs"

# storing results
results = []

# process through caption
for filename in os.listdir(custom_dir):
    if not filename.endswith("_caption.txt"):
        continue

    base = filename.replace("_caption.txt", "")
    caption_path = os.path.join(custom_dir, filename)
    custom_story_path = os.path.join(custom_dir, f"{base}_custom_story.txt")
    gpt4o_story_path = os.path.join(gpt4o_dir, f"{base}_gpt4o_story.txt")

    # Skip if story file is missing
    if not os.path.exists(custom_story_path) or not os.path.exists(gpt4o_story_path):
        continue

    # Read files
    with open(caption_path, "r", encoding="utf-8") as f:
        raw_caption = f.read().strip()

    with open(custom_story_path, "r", encoding="utf-8") as f:
        custom_story = f.read().strip()

    with open(gpt4o_story_path, "r", encoding="utf-8") as f:
        gpt4o_story = f.read().strip()

    # for cleaner caption
    if not raw_caption or len(raw_caption.split()) < 4:
        continue
    if "declaration of the declaration" in raw_caption.lower() or "tt tt" in raw_caption.lower():
        continue

    # using only first sentence of the caption
    caption = raw_caption.split('.')[0].strip() + '.'

    # bert score 
    try:
        P1, R1, F1_custom = bert_score(
            [custom_story], [caption], lang="en", rescale_with_baseline=False)
        P2, R2, F1_gpt4o = bert_score(
            [gpt4o_story], [caption], lang="en", rescale_with_baseline=False)
    except Exception as e:
        print(f"⚠️ Skipped {base} due to BERTScore error: {e}")
        continue

    results.append({
        "image": base,
        "caption": caption,
        "custom_alignment": round(F1_custom[0].item(), 4),
        "gpt4o_alignment": round(F1_gpt4o[0].item(), 4)
    })

# save as CSV
df = pd.DataFrame(results)
df.to_csv("caption_alignment_scores.csv", index=False)
print("✅ Saved cleaned alignment scores to 'caption_alignment_scores.csv'")
