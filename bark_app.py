from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_1" # please change to en_speaker_6 for male voice 

inputs = processor("In the heart of the bustling city, where the sounds of traffic blend with the laughter of children playing in the park, a small café stands quietly, inviting passersby with the rich aroma of freshly brewed coffee and baked pastries; it is a place where stories are shared, friendships are formed, and time seems to slow down just enough for one to savor the simple pleasures of life", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)