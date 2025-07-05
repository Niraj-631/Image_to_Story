import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
import time
from streamlit_lottie import st_lottie
import requests
import io
#import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Expanded language options for gTTS
LANG_OPTIONS = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Hindi": "hi",
    "Arabic": "ar"
}

@st.cache_resource
def load_blip_model():
    """Load BLIP model with caching for better performance"""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
        return processor, model
    except Exception as e:
        logger.error(f"Error loading BLIP model: {e}")
        st.error("Failed to load image captioning model. Please check your internet connection.")
        return None, None

@st.cache_resource
def load_t5_model():
    """Load T5 model with caching for better performance"""
    try:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to("cpu")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading T5 model: {e}")
        st.error("Failed to load story generation model. Please check your internet connection.")
        return None, None

def generate_caption_blip(image, processor, model):
    """Generate caption from image using BLIP model"""
    try:
        if processor is None or model is None:
            return "Unable to generate caption - model not loaded"
        
        image_rgb = image.convert('RGB')
        inputs = processor(image_rgb, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=4)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return "Unable to generate caption from image"

def generate_story_t5(prompt, tokenizer, model, story_length="medium"):
    """Generate story using T5 model with different length options"""
    try:
        if tokenizer is None or model is None:
            return "Unable to generate story - model not loaded"
        
        # Adjust max_length based on story length preference
        length_map = {"short": 100, "medium": 200, "long": 300}
        max_length = length_map.get(story_length, 200)
        
        # Enhanced prompt for better story generation
        enhanced_prompt = f"Write a creative short story about: {prompt}"
        
        input_ids = tokenizer.encode(enhanced_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_length=max_length, 
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        story = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return story
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        return "Unable to generate story from the image description"

def text_to_speech(story, lang='en'):
    """Convert text to speech and return audio bytes"""
    try:
        if not story or len(story.strip()) == 0:
            return None
        
        tts = gTTS(text=story, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return None

def load_lottie_url(url):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.error(f"Error loading Lottie animation: {e}")
    return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="🎨 Image-to-Story Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for enhanced UI
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .block-container {
            padding: 2rem;
            max-width: 1200px;
        }
        .title-text {
            text-align: center;
            color: #ffffff;
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle-text {
            text-align: center;
            color: #e0e6ff;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .caption-box {
            background: rgba(255, 255, 255, 0.1);
            border-left: 6px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            font-size: 1.1rem;
            color: #ffffff;
            margin: 20px 0;
            backdrop-filter: blur(10px);
        }
        .story-box {
            background: rgba(255, 255, 255, 0.1);
            border-left: 6px solid #FF6B6B;
            border-radius: 10px;
            padding: 20px;
            font-size: 1.1rem;
            color: #ffffff;
            margin: 20px 0;
            backdrop-filter: blur(10px);
            line-height: 1.6;
        }
        .sidebar .sidebar-content {
            background: rgba(255, 255, 255, 0.1);
        }
        .stButton > button {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: bold;
            font-size: 1.1rem;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title
    st.markdown("<h1 class='title-text'>🎨 Image-to-Story Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Transform your images into captivating stories with AI-powered creativity</p>", unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2485/2485634.png", width=100)
        st.markdown("### 🎛️ Configuration")
        
        language = st.selectbox("🌐 Audio Language:", list(LANG_OPTIONS.keys()))
        story_length = st.selectbox("📏 Story Length:", ["short", "medium", "long"])
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Upload clear, high-quality images
        - Images with people, objects, or scenes work best
        - Try different story lengths for variety
        - Use different languages for audio output
        """)

    # Load models
    with st.spinner("Loading AI models..."):
        blip_processor, blip_model = load_blip_model()
        t5_tokenizer, t5_model = load_t5_model()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload an image to generate a story from"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

            # Generate story button
            if st.button("✨ Generate Story", key="gen_btn"):
                start_time = time.time()

                with st.spinner("AI is crafting your story..."):
                    # Load animation
                    lottie_processing = load_lottie_url("https://lottie.host/1aa4b60f-7593-4a5c-8733-0f0b74fa1880/RtxqXolRya.json")
                    if lottie_processing:
                        st_lottie(lottie_processing, height=200, key="processing")

                    # Generate caption
                    caption = generate_caption_blip(image, blip_processor, blip_model)
                    
                    # Generate story
                    story = generate_story_t5(caption, t5_tokenizer, t5_model, story_length)
                    
                    # Generate audio
                    audio_bytes = text_to_speech(story, lang=LANG_OPTIONS[language])
                    
                    elapsed = time.time() - start_time

                st.success(f"✅ Story generated in {elapsed:.2f} seconds!")

                # Store results in session state
                st.session_state.caption = caption
                st.session_state.story = story
                st.session_state.audio_bytes = audio_bytes
                st.session_state.language = language

    with col2:
        st.markdown("### 📚 Generated Content")
        
        # Display results if available
        if hasattr(st.session_state, 'caption') and st.session_state.caption:
            st.markdown("#### 🏷️ Image Description")
            st.markdown(f"<div class='caption-box'>{st.session_state.caption}</div>", unsafe_allow_html=True)
            
            st.markdown("#### 📖 Your Story")
            st.markdown(f"<div class='story-box'>{st.session_state.story}</div>", unsafe_allow_html=True)
            
            # Audio playback
            if st.session_state.audio_bytes:
                st.markdown("#### 🔊 Listen to Your Story")
                st.audio(st.session_state.audio_bytes, format='audio/mp3')
            
            # Download options
            st.markdown("#### 📥 Download Options")
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                st.download_button(
                    "📄 Download Story",
                    st.session_state.story,
                    file_name="generated_story.txt",
                    mime="text/plain"
                )
            
            with col_download2:
                if st.session_state.audio_bytes:
                    st.download_button(
                        "🎵 Download Audio",
                        st.session_state.audio_bytes,
                        file_name="generated_story.mp3",
                        mime="audio/mpeg"
                    )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #e0e6ff; font-size: 0.9rem;'>"
        "Made with ❤️ using Streamlit, BLIP, T5, and gTTS"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()