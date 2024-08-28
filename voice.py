
from elevenlabs.client import ElevenLabs
from elevenlabs import play


client = ElevenLabs(api_key = "sk_d3d958530a1e2f38fa37e1c8f56e83824e606b95581f9d07")

audio = client.generate(
  text= "hello balaji..how are you",
  voice="Nicole",
  model="eleven_multilingual_v2"
)
play(audio)