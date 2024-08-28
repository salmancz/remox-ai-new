import openai
import pyaudio
import wave
import datetime
import audioop
import dotenv
from dotenv import load_dotenv, find_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv(find_dotenv())

messages = []

openai.api_key = "sk-03iQoZXnrS5ytJBvRCIT33JV6DQmU6VD7PBkjZfjoHT3BlbkFJGRkk8No9vtqKvBnLuEDTZpXxUx-QXFkrooV7kTOQQA"
elevenlabs_api_key = "2b658a0fd6e424c1986a73d2db6fd778"

client = ElevenLabs(api_key=elevenlabs_api_key)

def result(output):
    audio = client.generate(
        text=output,
        voice="Rachel",
        model="eleven_multilingual_v2"
    )
    play(audio)

def get_response_from_ai(input):
    global messages
    
    template = """
    You are an ai assitant for visually imparied people, you just tell what is infornt of them.  
    Situation: There is a man sitting with a 
    {history}
    human: {input}
    you: 
    """

    messages.append({"role": "user", "content": input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    response_text = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": response_text})
    
    result(response_text)

def record_audio(filename, silence_duration=2, max_duration=20, sample_rate=44100, channels=2, chunk_size=1024):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording...")
    frames = []
    silence_frames = 0  # Counter for consecutive frames of silence
    silence_threshold = 1000  # Adjust this threshold as needed
    recording_duration = 0

    while recording_duration < max_duration:
        data = stream.read(chunk_size)
        frames.append(data)

        # Check audio energy to detect silence
        rms = audioop.rms(data, 2)  # 2 for sample width of 16 bits (paInt16)
        
        if rms < silence_threshold:
            silence_frames += 1
        else:
            silence_frames = 0

        if silence_frames >= int(silence_duration * sample_rate / chunk_size):
            # Increment recording duration during silence
            recording_duration += silence_duration
        else:
            recording_duration += chunk_size / sample_rate

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

if __name__ == "__main__":
    while True:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"aud/audio_{timestamp}.wav"
        record_audio(filename)
        print(f"Audio saved as {filename}")

        audio_file = open(filename, "rb")
        transcript = openai.Audio.translate("whisper-1", audio_file)
        print(transcript.text)
        get_response_from_ai(transcript.text)
