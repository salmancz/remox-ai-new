from elevenlabs import play
from elevenlabs.client import ElevenLabs
from openai import OpenAI
import time




client1 = ElevenLabs(
  api_key="sk_7bfa6b249f3d6040ba26670152b02901c0263ae833de3b47", # Defaults to ELEVEN_API_KEY
)

client = OpenAI(api_key="sk-03iQoZXnrS5ytJBvRCIT33JV6DQmU6VD7PBkjZfjoHT3BlbkFJGRkk8No9vtqKvBnLuEDTZpXxUx-QXFkrooV7kTOQQA")
image_url = 'https://th-i.thgim.com/public/migration_catalog/article11103092.ece/alternates/FREE_1200/VBK-WALKING'

language = "Tamil"

st = time.time()

response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
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
                    "image_url": {"url": image_url}
                }
            ],
        }
    ],
    max_tokens=500,
)
result = response.choices[0].message.content
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

stt = time.time()

audio = client1.generate(
  text= result,
  voice="Charlie",
  model="eleven_multilingual_v2"
)

play(audio)

ett = time.time()
elapsedd_time = ett - stt
print('Execution time:', elapsedd_time, 'seconds')




