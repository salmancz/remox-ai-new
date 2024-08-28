import streamlit as st
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from openai import OpenAI
import time
import cv2
import numpy as np
from PIL import Image

# Initialize OpenAI and ElevenLabs clients
openai_client = OpenAI(api_key="sk-03iQoZXnrS5ytJBvRCIT33JV6DQmU6VD7PBkjZfjoHT3BlbkFJGRkk8No9vtqKvBnLuEDTZpXxUx-QXFkrooV7kTOQQA")
client1 = ElevenLabs(
  api_key="sk_7bfa6b249f3d6040ba26670152b02901c0263ae833de3b47", # Defaults to ELEVEN_API_KEY
)
# Set the language for the response
language = "Tamil"

# Streamlit app
st.title("Visually Impaired Smart Glasses AI")


st.write("Click the button below to capture an image from your camera:")

# Button to capture image
if st.button("Capture Image"):
    # Open the camera and capture an image
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    # Convert the image to a format suitable for OpenAI
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    st.image(img, caption="Captured Image", use_column_width=True)
    
    # Convert the image to a byte array
    img_byte_arr = np.array(img)
    _, img_encoded = cv2.imencode('.jpg', img_byte_arr)
    img_bytes = img_encoded.tobytes()

    # Use the captured image for processing
    st.write("Analyzing the image...")

    # Start time for OpenAI request
    start_time = time.time()

    # OpenAI request
    response = openai_client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "user",
                "content": f"""
                    I’m a visually impaired person and you are an AI in my visually impaired smart glasses. You have some rules which you should follow:
                    1. Your job is to help me by being my eyes.
                    2. Your primary job is to assist me and help me to navigate, identify, read, and tell me things.
                    3. I’m a native {language} speaker, so the response should be in {language}, but not in grammatical {language}. I may need some English words in between and some uh... like sounds to make it real too.
                    4. I want a scenario to be explained by an image that I pass to you. You’ll be analyzing the image, and you should tell me what you see.
                    5. Don't start with "in this image" or similar. Keep it simple and make sure to say everything in 1 to 2 sentences.
                    6. It would be great if you could say the distance of some very specific things, obstacles, or a gateway/door near me. (just be casual)
                    7. You should make sure every rule above is satisfied.
                """
            },
            {
                "role": "system",
                "content": "Here is the image for analysis.",
                "image": {"url": img_bytes}
            }
        ],
        max_tokens=500,
    )

    # Get the response content
    result = response.choices[0].message.content

    # End time for OpenAI request
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"OpenAI Processing Time: {elapsed_time:.2f} seconds")

    # Generate speech using ElevenLabs
    st.write("Generating audio...")

    audio_start_time = time.time()

    audio = client1.generate(
    text= result,
    voice="Charlie",
    model="eleven_multilingual_v2")

    audio_end_time = time.time()
    audio_elapsed_time = audio_end_time - audio_start_time
    st.write(f"Audio Generation Time: {audio_elapsed_time:.2f} seconds")

    # Play the audio
    play(audio)

    st.success("Process completed successfully!")

