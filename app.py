import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
#import os
import time
from streamlit_lottie import st_lottie
import requests

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

# Load T5
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

# Languages for gTTS
LANG_OPTIONS = {
    "English": "en"
}

def generate_caption_blip(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(image, return_tensors="pt").to(DEVICE)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_story_t5(prompt):
    input_ids = t5_tokenizer.encode("tell a story: " + prompt, return_tensors="pt").to(DEVICE)
    output_ids = t5_model.generate(input_ids, max_length=150, num_return_sequences=1)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def text_to_speech(story, filename="story.mp3", lang='en'):
    tts = gTTS(text=story, lang=lang)
    tts.save(filename)
    return filename

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
lottie_processing = load_lottie_url("https://lottie.host/1aa4b60f-7593-4a5c-8733-0f0b74fa1880/RtxqXolRya.json")

# Streamlit UI
st.set_page_config(page_title="Image-to-Story Generator", layout="centered")

# Custom CSS for advanced UI
st.markdown("""
    <style>
    .main {
        background-color: #f3f5fa;
    }
    .block-container {
        padding: 2rem;
    }
    .title-text {
        text-align: center;
        color: #003566;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .caption-box {
        background-color: #e8f0fe;
        border-left: 6px solid #3b82f6;
        border-radius: 8px;
        padding: 15px;
        font-size: 1.1rem;
        color: #111827;
        margin-top: 15px;
    }
    .story-button button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-text'> Image-to-Story Generator</h1>", unsafe_allow_html=True)
st.markdown("Transform your uploaded image into a creative short story, complete with audio in your language.")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2485/2485634.png", width=100)
    language = st.selectbox("🌐 Select Audio Language:", list(LANG_OPTIONS.keys()))

uploaded_file = st.file_uploader("📤 Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())
    st.image(image_path, caption="🖼️ Uploaded Image", use_column_width=True)

    if st.button("✨ Generate Story", key="gen_btn"):
        start_time = time.time()

        with st.spinner("AI is generating your story..."):
            if lottie_processing:
                st_lottie(lottie_processing, height=200, key="processing")
            else:
                st.info("Generating story... please wait ⏳")

            caption = generate_caption_blip(image_path)
            story = generate_story_t5(caption)
            audio_file = text_to_speech(story, lang=LANG_OPTIONS[language])
            elapsed = time.time() - start_time

        st.success(f"✅ Story generated in {elapsed:.2f} seconds!")

        st.markdown("### 📝 Generated Story")
        st.markdown(f"<div class='caption-box'>{story}</div>", unsafe_allow_html=True)

        st.markdown("### 🔊 Audio Story")
        st.audio(audio_file)

        st.download_button("📄 Download Story as Text", story, file_name="story.txt")
        with open(audio_file, "rb") as f:
            st.download_button("🔊 Download Audio", f, file_name="story.mp3")
