import os
import streamlit as st
from dotenv import load_dotenv
from textwrap import dedent
from youtube_transcript_api import YouTubeTranscriptApi

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.youtube import YouTubeTools

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY is missing in your .env file.")
    st.stop()

# ----------------------------
# Initialize AI Agent
# ----------------------------
agent = Agent(
    model=Groq(id="llama-3.1-8b-instant", api_key=groq_api_key),
    tools=[YouTubeTools()],
    description="A YouTube video analyzer and summarizer.",
    instructions=dedent("""
        You are a YouTube content analyst.
        Given a YouTube URL:
        1. Fetch video captions and metadata.
        2. Provide a structured summary with key points.
        3. Optionally include timestamps for important sections.
    """),
    show_tool_calls=True,
    markdown=True,
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="YouTube Summarizer", page_icon="🎬", layout="centered")
st.markdown("""
    <div style="text-align:center;">
        <h1>🎬 Agno YouTube Summarizer</h1>
        <p>Paste a YouTube URL to generate an AI-powered summary.</p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# User Input Form
# ----------------------------
with st.form(key="url_form"):
    youtube_url = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    submit = st.form_submit_button("Summarize Video")

# ----------------------------
# Helper Functions
# ----------------------------
def extract_video_id(url: str) -> str:
    """Extracts the video ID from a YouTube URL."""
    return url.split("v=")[-1].split("&")[0]

def summarize_with_transcript(video_id: str) -> str:
    """Fetch transcript and summarize using the AI agent."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([t["text"] for t in transcript])
    result = agent.run(f"Summarize the following transcript:\n\n{transcript_text}")
    return result.content if hasattr(result, "content") else str(result)

# ----------------------------
# Main Processing Logic
# ----------------------------
if submit and youtube_url:
    with st.spinner("⏳ Generating summary…"):
        try:
            # 1️⃣ Try Agno's built-in summarization
            response = agent.run(f"Summarize this video: {youtube_url}")
            summary_text = response.content if hasattr(response, "content") else str(response)

            # 2️⃣ If captions are missing, use direct transcript API
            if not summary_text or "couldn't retrieve" in summary_text.lower():
                try:
                    video_id = extract_video_id(youtube_url)
                    summary_text = summarize_with_transcript(video_id)
                except Exception as transcript_error:
                    st.warning(f"⚠ Could not fetch transcript: {transcript_error}")

            # 3️⃣ Display result
            if summary_text:
                st.markdown("### 📄 Summary")
                st.markdown(summary_text)
            else:
                st.warning("No summary found. The video may not have captions.")

        except Exception as e:
            st.error(f"⚠ Something went wrong: {e}")
            st.info("Please check your API key or internet connection.")
