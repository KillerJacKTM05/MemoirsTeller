import os
import google.generativeai as genai

# load gemini key
genai.configure(api_key="AIzaSyAZEiksFHyBNmjdLXBwZVB_TcEzlKqprx0")

input_folder = "test_images"
output_folder = "gemini_outputs"
os.makedirs(output_folder, exist_ok=True)

# default prompt settings
style = "Jane Austen (Romantic Wit)"
mood = "Romantic and Dreamy"
time_period = "Victorian Era (1837-1901)"
characters = ""  # Gemini invents 
length = "Long (500-600 words)"

# Gemini model
#model = genai.GenerativeModel("gemini-pro-vision")
model = genai.GenerativeModel("gemini-1.5-pro")

# Building prompt
def build_prompt(style, mood, time_period, characters, length):
    return f"""You are a masterful storyteller writing in the style of {style}.

Scene to base your story on: Use the image directly (do not assume a caption).

Parameters:
- Mood/Tone: {mood}
- Time Period: {time_period}
- Characters and people in the image: {characters or "Create original characters as needed"}
- Length: {length}

Write a compelling story that:
1. Captures the essence of the visual scene
2. Maintains the specified mood and literary style
3. Incorporates the time period naturally
4. Creates vivid, immersive descriptions
5. Has a clear narrative arc
6. Uses the character information provided to accurately represent the people in the image

Make it feel like a genuine diary entry or memoir excerpt with emotional depth.
"""

# Generate story
from PIL import Image


for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, filename)
    print(f"📸 Processing: {filename}")

    try:
        image = Image.open(image_path)
        prompt = build_prompt(style, mood, time_period, characters, length)

        response = model.generate_content([prompt, image])
        story = response.text.strip()

        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_folder, f"{base_name}_gemini_story.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(story)

        print(f"✅ Saved: {out_path}")

    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")

print("\n🎉 Done! All Gemini stories saved to 'gemini_outputs/'")
