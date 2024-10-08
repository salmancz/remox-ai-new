import cv2
import streamlit as st
from PIL import Image
import numpy as np
import base64
import requests
import time
from elevenlabs import play
from elevenlabs.client import ElevenLabs


client1 = ElevenLabs(
  api_key="sk_7bfa6b249f3d6040ba26670152b02901c0263ae833de3b47", # Defaults to ELEVEN_API_KEY
)
# Set the language for the response
language = "Tamil"

# Function to encode the image to base64
def encode_image(image):
    img_byte_arr = np.array(image)
    _, img_encoded = cv2.imencode('.jpg', img_byte_arr)
    return base64.b64encode(img_encoded.tobytes()).decode('utf-8')

# OpenAI API Key
api_key = "sk-03iQoZXnrS5ytJBvRCIT33JV6DQmU6VD7PBkjZfjoHT3BlbkFJGRkk8No9vtqKvBnLuEDTZpXxUx-QXFkrooV7kTOQQA"

# Streamlit app
st.title("Image Capture and OpenAI Analysis")

# Button to capture the image
if st.button("Capture Image"):
    # Open the camera and capture an image
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert the captured image to RGB format for display
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Display the captured image in the Streamlit app
        st.image(img, caption="Captured Image", use_column_width=True)

        # Convert the image to base64 format
        base64_image = encode_image(img)

        # Prepare the API request payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",  # Ensure you use the correct model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
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
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        # Start time for OpenAI request
        start_time = time.time()

        # Make the API request
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        # End time for OpenAI request
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Check if the request was successful
        if response.status_code == 200:
            # Convert the response to JSON
            result = response.json()
            # Extract and display the content
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                st.write(content)
                
                audio = client1.generate(
                text= content,
                voice="Charlie",
                model="eleven_multilingual_v2")
                play(audio)

            else:
                st.write("No choices found in the response.")
            st.write(f"OpenAI Processing Time: {elapsed_time:.2f} seconds")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.error("Failed to capture image. Please try again.")
