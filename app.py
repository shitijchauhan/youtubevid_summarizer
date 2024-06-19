import os
import openai
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

# Specify the path to your .env file
env_path = r'c:/Users/shiti/OneDrive/Documents/youtube_summarizer[1]/youtube_summarizer/.env/openai_api' 

# Load the OpenAI API key from the .env file
load_dotenv(env_path)
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Try fetching the manual transcript
    try:
        transcript = transcript_list.find_manually_created_transcript()
        language_code = transcript.language_code  # Save the detected language
    except:
        # If no manual transcript is found, try fetching an auto-generated transcript in a supported language
        try:
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            transcript = generated_transcripts[0]
            language_code = transcript.language_code  # Save the detected language
        except:
            # If no auto-generated transcript is found, raise an exception
            raise Exception("No suitable transcript found.")

    full_transcript = " ".join([part['text'] for part in transcript.fetch()])
    return full_transcript, language_code  # Return both the transcript and detected language


def summarize_with_langchain_and_openai(transcript, language_code, model_name='gpt-3.5-turbo'):
    # Split the document if it's too long
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript)
    text_to_summarize = " ".join(texts[:4]) # Adjust this as needed

    # Prepare the prompt for summarization
    system_prompt = 'I want you to act as a Life Coach that can create good summaries!'
    prompt = f'''Summarize the following text in {language_code}.
    Text: {text_to_summarize}

    Add a title to the summary in {language_code}. 
    Include an INTRODUCTION, BULLET POINTS if possible, and a CONCLUSION in {language_code}.'''

    # Start summarizing using OpenAI
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        temperature=1
    )
    
    return response['choices'][0]['message']['content']

def main():
    st.title('YouTube video summarizer')
    link = st.text_input('Enter the link of the YouTube video you want to summarize:')

    if st.button('Start'):
        if link:
            try:
                progress = st.progress(0)
                status_text = st.empty()

                status_text.text('Loading the transcript...')
                progress.progress(25)

                # Getting both the transcript and language_code
                transcript, language_code = get_transcript(link)

                status_text.text(f'Creating summary...')
                
                summary_text = """
Introduction:
The iPhone 15 Pro and 15 Pro Max are introduced as phones without standout features 
initially but reveal surprising benefits upon extended use.

Unboxing Experience:
The unboxing experience includes the new woven fiber USBC cable, which is more 
durable, but not a fast cable for USB 3 speeds.

Action Button:
The new programmable action button is discussed, highlighting its versatility and 
usefulness for camera shortcuts.

Call Quality and Voice Isolation:
Improved voice isolation feature enhances call quality, especially in noisy 
environments.

Performance:
Performance benchmarks, CPU, and graphics performance compared to previous models, 
including gaming capabilities.

Design and Build:
Changes in design, materials, and comfort compared to previous models, including 
durability and aesthetic aspects.

Camera System:
Detailed breakdown of the camera system, including megapixel count, file formats, 
portrait mode, and processing improvements.

Zoom and Optical Features:
Discussion of zoom capabilities, optical vs. digital zoom, and the quality of 
zoomed-in shots.

Battery Life and Charging:
Comparison of battery performance to previous models, charging speeds, and USBC 
capabilities.

Price and Conclusion:
Evaluation of pricing, comparisons between iPhone 15 Pro and Pro Max, and final 
thoughts on the devices.
"""

                status_text.text(summary_text)

                progress.progress(75)

                model_name = 'gpt-3.5-turbo'
                summary = summarize_with_langchain_and_openai(transcript, language_code, model_name)

                status_text.text('Summary:')
                st.markdown(summary)
                progress.progress(100)
            except Exception as e:
                st.write(str(e))
        else:
            st.write('Please enter a valid YouTube link.')

if __name__ == "__main__":
    main()
