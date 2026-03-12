import gradio as gr
import whisper
from transformers import pipeline

# Load models
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded!")

print("Loading sentiment model...")
sentiment = pipeline("sentiment-analysis")
print("Sentiment model loaded!")

# Function
def speech_sentiment(audio_file):

    if audio_file is None:
        return "No audio uploaded", "No sentiment"

    # Transcribe audio
    result = model.transcribe(audio_file)
    text = result["text"]

    # Sentiment analysis
    sentiment_result = sentiment(text)

    label = sentiment_result[0]["label"]
    score = sentiment_result[0]["score"]

    sentiment_output = f"{label} ({score:.2f})"

    return text, sentiment_output


# Gradio Interface
iface = gr.Interface(
    fn=speech_sentiment,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="Sentiment Result")
    ],
    title="Speech Sentiment Analysis",
    description="Upload an audio file to get transcription and sentiment analysis."
)

if __name__ == "__main__":
    iface.launch()
