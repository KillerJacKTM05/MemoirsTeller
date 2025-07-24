import os
from PIL import Image
from Teller import memoirs_pipeline

# Folders
input_folder = "test_images"
output_folder = "custom_outputs"
os.makedirs(output_folder, exist_ok=True)

# default settings
style = "Jane Austen (Romantic Wit)"
mood = "Romantic and Dreamy"
time_period = "Victorian Era (1837-1901)"
characters = ""  # let the model generate
length = "Long (500-600 words)"
detailed_caption = True  # if detailed generation wanted

# Loop through images
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, filename)
    image = Image.open(image_path).convert("RGB")

    print(f"🖼️ Processing with custom model: {filename}")

    # Running our pipeline
    caption, story, _ = memoirs_pipeline(
        image=image,
        style=style,
        character_info=characters,
        mood=mood,
        length=length,
        setting_time=time_period,
        detailed_caption=detailed_caption
    )

    # Saving the output
    base_name = os.path.splitext(filename)[0]

    with open(os.path.join(output_folder, f"{base_name}_caption.txt"), "w", encoding="utf-8") as f:
        f.write(caption)

    with open(os.path.join(output_folder, f"{base_name}_custom_story.txt"), "w", encoding="utf-8") as f:
        f.write(story)

    print(f"✅ Saved: {base_name}_custom_story.txt")

print("\n🎉 Done! All custom model stories saved in 'custom_outputs/'")
