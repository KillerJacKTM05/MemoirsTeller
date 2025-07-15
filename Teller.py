import os
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import openai
from datetime import datetime

# Try loading from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
# Load BLIP for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Caption generation
def generate_caption(image: Image.Image, detailed: bool = False):
    if image is None:
        return "No image provided."
    
    # Basic caption
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=50, num_beams=5)
    basic_caption = processor.decode(output[0], skip_special_tokens=True)
    
    if not detailed:
        return basic_caption
    
    # Generate multiple captions with different prompts
    detailed_prompts = [
        "a detailed description of",
        "the scene shows",
        "in this image we can see",
        "the photograph captures"
    ]
    
    detailed_parts = []
    for prompt in detailed_prompts:
        try:
            inputs = processor(
                images=image, 
                text=prompt,
                return_tensors="pt"
            ).to(device)
            output = model.generate(**inputs, max_length=80, num_beams=3, temperature=0.7)
            caption_part = processor.decode(output[0], skip_special_tokens=True)
            if caption_part and caption_part != prompt:
                detailed_parts.append(caption_part)
        except:
            continue
    
    # Combine basic caption with detailed parts
    if detailed_parts:
        enhanced_caption = f"{basic_caption.capitalize()}. "
        enhanced_caption += " ".join(detailed_parts[:2])  # Use first 2 detailed parts
        return enhanced_caption
    
    return basic_caption

def generate_story(caption: str, style: str, character_info: str, mood: str, length: str, setting_time: str):
    # Map length to token counts
    length_mapping = {
        "Short (100-200 words)": 250,
        "Medium (300-400 words)": 500,
        "Long (500-600 words)": 750
    }
    max_tokens = length_mapping.get(length, 500)
    
    # Prompt for more context
    prompt = f"""You are a masterful storyteller writing in the style of {style}.

Scene to base your story on: "{caption}"

Parameters:
- Mood/Tone: {mood}
- Time Period: {setting_time}
- Characters and people in the image: {character_info if character_info else "Create original characters as needed"}
- Length: {length}

Write a compelling story that:
1. Captures the essence of the visual scene
2. Maintains the specified mood and literary style
3. Incorporates the time period naturally
4. Creates vivid, immersive descriptions
5. Has a clear narrative arc
6. Uses the character information provided to accurately represent the people in the image

Make it feel like a genuine diary entry or memoir excerpt with emotional depth."""

    # List of models to try in order of preference
    models_to_try = [
        "gpt-4o",           
        "gpt-4o-mini",      
        "gpt-4o-nano",      
        "gpt-4.1-mini",
        "gpt-3.5-turbo",    
        "o1-mini",
        "gpt2" # smallest
    ]
    
    client = openai.OpenAI(api_key=openai.api_key)
    
    for model in models_to_try:
        try:
            print(f"Trying model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=max_tokens,
                top_p=0.9
            )
            print(f"Success with model: {model}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Model {model} failed: {str(e)}")
            continue
    
    return "Error: Could not generate story with any available model."


# Save story to file
def save_story(story_text: str, caption: str, metadata: dict):
    if not story_text or story_text.startswith("Error"):
        return "Cannot save - no valid story generated."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"memoir_{timestamp}.txt"
    
    try:
        os.makedirs("saved_stories", exist_ok=True)
        filepath = os.path.join("saved_stories", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Memoir Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Scene Description: {caption}\n\n")
            f.write(f"Style: {metadata.get('style', 'Unknown')}\n")
            f.write(f"Mood: {metadata.get('mood', 'Unknown')}\n")
            f.write(f"Time Period: {metadata.get('time_period', 'Unknown')}\n")
            f.write(f"Characters: {metadata.get('characters', 'Either Not exist or not specified')}\n")
            f.write(f"Length: {metadata.get('length', 'Unknown')}\n\n")
            f.write("Story:\n")
            f.write("-" * 20 + "\n")
            f.write(story_text)
        
        return f"Story saved as {filename}"
    except Exception as e:
        return f"Error saving story: {str(e)}"

# Main
def memoirs_pipeline(image, style, character_info, mood, length, setting_time, detailed_caption):
    if image is None:
        return "Please upload an image first.", "", {}
    
    caption = generate_caption(image, detailed=detailed_caption)
    story = generate_story(caption, style, character_info, mood, length, setting_time)
    
    # Prepare metadata for saving
    metadata = {
        'style': style,
        'mood': mood,
        'time_period': setting_time,
        'characters': character_info if character_info else "Either Not exist or not specified",
        'length': length
    }
    
    return caption, story, metadata
    
# Save for the button
def save_current_story(story, caption, style, mood, setting_time, character_info, length):
    if not story or story.startswith("Error"):
        return "No valid story to save."
    
    metadata = {
        'style': style,
        'mood': mood,
        'time_period': setting_time,
        'characters': character_info if character_info else "Either Not exist or not specified",
        'length': length
    }
    return save_story(story, caption, metadata)

# Options for storytelling
literary_styles = [
    "Edgar Allan Poe (Gothic Horror)",
    "Jane Austen (Romantic Wit)",
    "H.P. Lovecraft (Cosmic Horror)",
    "J.R.R. Tolkien (Epic Fantasy)",
    "Ray Bradbury (Sci-Fi Mystery)",
    "Ernest Hemingway (Minimalist)",
    "Virginia Woolf (Stream of Consciousness)",
    "Gabriel García Márquez (Magical Realism)",
    "Your Own Diary Voice"
]

mood_options = [
    "Nostalgic and Melancholic",
    "Mysterious and Intriguing",
    "Joyful and Uplifting",
    "Dark and Brooding",
    "Romantic and Dreamy",
    "Adventurous and Exciting",
    "Peaceful and Serene",
    "Humorous and Light-hearted"
]

length_options = [
    "Short (100-200 words)",
    "Medium (300-400 words)",
    "Long (500-600 words)"
]

time_periods = [
    "Present Day",
    "Victorian Era (1837-1901)",
    "1920s Jazz Age",
    "Medieval Times",
    "Future/Sci-Fi",
    "Ancient Times",
    "1950s Post-War",
    "Wild West (1800s)"
]

# Gradio UI
with gr.Blocks(title="Memoirs Teller - Story Creator Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📝 Memoirs Teller
    ### Transform your images into captivating stories
    Upload an image and watch as your agent creates a unique story in your chosen literary style.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📸 Upload Your Image")
            detailed_caption = gr.Checkbox(label="Generate detailed scene description", value=False)
            
        with gr.Column(scale=1):
            style_input = gr.Dropdown(
                choices=literary_styles, 
                label="📚 Literary Style", 
                value="Edgar Allan Poe (Gothic Horror)"
            )
            mood_input = gr.Dropdown(
                choices=mood_options,
                label="🎭 Story Mood",
                value="Nostalgic and Melancholic"
            )
            length_input = gr.Dropdown(
                choices=length_options,
                label="📏 Story Length",
                value="Medium (300-400 words)"
            )
            time_input = gr.Dropdown(
                choices=time_periods,
                label="⏰ Time Period",
                value="Present Day"
            )

    character_input = gr.Textbox(
        label="👥 If exists, describe People in the Photo",
        placeholder="Examples:\n• From left to right: Ann (my sister), Claus (our dog), Clara (grandmother)\n• The woman with the umbrella is Clara, age 74\n• Three children playing: Emma, Kurt, and Sophie\n• My father John and his American Muscle\n• Leave empty if not exists",
        lines=4,
        info="Please be briefly specific about who's in the photo - names, relationships, positions, or distinctive features"
    )

    with gr.Row():
        generate_btn = gr.Button("✨ Generate Story", variant="primary", size="lg")
        save_btn = gr.Button("💾 Save Story", variant="secondary")

    with gr.Row():
        with gr.Column():
            caption_output = gr.Textbox(label="🔍 Scene Analysis", lines=3)
        with gr.Column():
            save_status = gr.Textbox(label="💾 Save Status", lines=1)

    story_output = gr.Textbox(label="📖 Your Generated Story", lines=15)

    # Store metadata for saving
    metadata_state = gr.State()

    # Event handlers
    def generate_and_store(*args):
        caption, story, metadata = memoirs_pipeline(*args)
        return caption, story, metadata

    generate_btn.click(
        fn=generate_and_store,
        inputs=[
            image_input,
            style_input,
            character_input,
            mood_input,
            length_input,
            time_input,
            detailed_caption
        ],
        outputs=[caption_output, story_output, metadata_state]
    )

    save_btn.click(
        fn=save_current_story,
        inputs=[
            story_output, 
            caption_output, 
            style_input, 
            mood_input, 
            time_input, 
            character_input,  
            length_input
        ],
        outputs=[save_status]
    )
    
    # Examples and tips
    gr.Markdown("### 💡 Tips for better stories:")
    gr.Markdown("""
    - **Character Descriptions**: Be specific about who's in the photo. Use formats like:
      - "From left to right: Ann (my sister), Claus (our dog)"
      - "The woman with red dress is Clara, my grandmother"
      - "Three children: Emma (8), Luke (7), Sophie (9)"
    - **Relationships**: Include how people relate to each other or to you
    - **Distinctive Features**: Mention clothing, objects, or expressions that stand out
    - **Detailed Captions**: Enable for more descriptive scene analysis
    - **Style Matching**: Choose a mood that complements your literary style
    - **Time Period**: Historical settings can add rich context to your story
    """)

    gr.Markdown("### 📝 Character Description Examples:")
    gr.Markdown("""
    **Good Examples:**
    - "From left to right: Anna (my sister in the blue dress), Max (our golden retriever), Clara (grandmother with the walking stick)"
    - "The man in the suit is my father John, standing next to his 1965 Mustang"
    - "Three children playing in the garden: Emma (the tallest), Luke (middle), and Sophie (smallest, with pigtails)"
    - "Wedding photo: bride Sarah, groom Michael, and their families"
    
    **Simple Examples:**
    - "Anna and Max"
    - "My grandmother Clara"
    - "Three friends at the beach"
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)