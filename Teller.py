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

# Enhanced caption generation with better prompting
def generate_caption(image: Image.Image, detailed: bool = False):
    if image is None:
        return "No image provided."
    
    # Basic caption
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=50, num_beams=5)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    # Optional detailed description using conditional generation
    if detailed:
        inputs = processor(images=image, text="a detailed description of", return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=100, num_beams=5)
        detailed_caption = processor.decode(output[0], skip_special_tokens=True)
        return f"{caption}. {detailed_caption}"
    
    return caption

# Enhanced story generation with better prompt engineering
def generate_story(caption: str, style: str, names: str, mood: str, length: str, setting_time: str):
    # Map length to token counts
    length_mapping = {
        "Short (100-200 words)": 250,
        "Medium (300-400 words)": 500,
        "Long (500-600 words)": 750
    }
    max_tokens = length_mapping.get(length, 500)
    
    # Enhanced prompt with more context
    prompt = f"""You are a masterful storyteller writing in the style of {style}.

Scene to base your story on: "{caption}"

Story Parameters:
- Mood/Tone: {mood}
- Time Period: {setting_time}
- Characters to include: {names if names else "Create original characters as needed"}
- Length: {length}

Write a compelling story that:
1. Captures the essence of the visual scene
2. Maintains the specified mood and literary style
3. Incorporates the time period naturally
4. Creates vivid, immersive descriptions
5. Has a clear narrative arc

Make it feel like a genuine diary entry or memoir excerpt with emotional depth."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,  # Slightly lower for more coherent stories
            max_tokens=max_tokens,
            top_p=0.9
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating story: {str(e)}"

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
            f.write(f"Characters: {metadata.get('characters', 'None specified')}\n\n")
            f.write("Story:\n")
            f.write("-" * 20 + "\n")
            f.write(story_text)
        
        return f"Story saved as {filename}"
    except Exception as e:
        return f"Error saving story: {str(e)}"

# Main pipeline
def memoirs_pipeline(image, style, names, mood, length, setting_time, detailed_caption):
    if image is None:
        return "Please upload an image first.", "", "No story to save."
    
    caption = generate_caption(image, detailed=detailed_caption)
    story = generate_story(caption, style, names, mood, length, setting_time)
    
    # Prepare metadata for saving
    metadata = {
        'style': style,
        'mood': mood,
        'time_period': setting_time,
        'characters': names,
        'length': length
    }
    
    return caption, story, metadata

# Save function for the button
def save_current_story(story, caption, style, mood, setting_time, names):
    if not story or story.startswith("Error"):
        return "No valid story to save."
    
    metadata = {
        'style': style,
        'mood': mood,
        'time_period': setting_time,
        'characters': names
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
with gr.Blocks(title="Memoirs Teller - AI Story Creator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📝 Memoirs Teller
    ### Transform your images into captivating stories
    Upload an image and watch as AI creates a personalized story in your chosen literary style.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📸 Upload Your Image")
            detailed_caption = gr.Checkbox(label="Generate detailed scene description", value=False)
            
        with gr.Column(scale=1):
            style_input = gr.Dropdown(
                choices=literary_styles, 
                label="Literary Style", 
                value="Edgar Allan Poe (Gothic Horror)"
            )
            mood_input = gr.Dropdown(
                choices=mood_options,
                label="Story Mood",
                value="Nostalgic and Melancholic"
            )
            length_input = gr.Dropdown(
                choices=length_options,
                label="Story Length",
                value="Medium (300-400 words)"
            )
            time_input = gr.Dropdown(
                choices=time_periods,
                label="Time Period",
                value="Present Day"
            )
            names_input = gr.Textbox(
                label="Character Names", 
                placeholder="e.g. Anna, Elias, the old cat Whiskers",
                lines=2
            )

    with gr.Row():
        generate_btn = gr.Button("✨ Generate Story", variant="primary", size="lg")
        save_btn = gr.Button("💾 Save Story", variant="secondary")

    with gr.Row():
        with gr.Column():
            caption_output = gr.Textbox(label="Scene Analysis", lines=3)
        with gr.Column():
            save_status = gr.Textbox(label="Save Status", lines=1)

    story_output = gr.Textbox(label="Your Generated Story", lines=15)

    # Store metadata for saving
    metadata_state = gr.State()

    # Event handlers
    def generate_and_store(*args):
        caption, story, metadata = memoirs_pipeline(*args)
        return caption, story, metadata

    generate_btn.click(
        fn=generate_and_store,
        inputs=[image_input, style_input, names_input, mood_input, length_input, time_input, detailed_caption],
        outputs=[caption_output, story_output, metadata_state]
    )

    save_btn.click(
        fn=save_current_story,
        inputs=[story_output, caption_output, style_input, mood_input, time_input, names_input],
        outputs=[save_status]
    )

    # Add examples
    gr.Markdown("### Tips for better stories:")
    gr.Markdown("""
    - **Character Names**: Include specific names or relationships (e.g., "my grandmother Clara", "Detective Morrison")
    - **Detailed Captions**: Enable for more descriptive scene analysis
    - **Style Matching**: Choose a mood that complements your literary style
    - **Time Period**: Historical settings can add rich context to your story
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)