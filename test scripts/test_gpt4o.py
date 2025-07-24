import openai
import base64
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set.")

def generate_story_with_gpt4o(image_path, style, mood, time_period, characters, length):
    # reading and encoding of the image
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    # determining image MIME type
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    image_url = f"data:{mime_type};base64,{base64_image}"

    # using the same prompt structure as the main pipeline
    prompt = f"""You are a masterful storyteller writing in the style of {style}.

Scene to base your story on: Use the image directly (do not assume a caption).

Parameters:
- Mood/Tone: {mood}
- Time Period: {time_period}
- Characters and people in the image: {characters if characters else "Create original characters as needed"}
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

    # call gpt4o with image and the prompt
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.strip()},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0.8,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[GPT-4o Error] {str(e)}"

story = generate_story_with_gpt4o(
    image_path="test_images/photo1.jpg",
    style="Jane Austen (Romantic Wit)",
    mood="Romantic and Dreamy",
    time_period="Victorian Era (1837-1901)",
    characters="",  # you can also add description like "x person is walking through the ... "
    length="Long (500-600 words)"
    #detailed_caption = True #enable if you want the detailed generation
)

print(story)
