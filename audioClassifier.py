import torch
import numpy as np
import librosa
from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from datasets import Audio
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_name = "distil-whisper/distil-large-v3"  
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Load the pre-trained Whisper model, adjust the dir relatively
model = WhisperForAudioClassification.from_pretrained(
    "models/whisper-large-v3_ADReSSo/checkpoint-93", num_labels=2, ignore_mismatched_sizes=True
)
model.to(device)
model.eval()

# Function to preprocess a single audio file
def preprocess_single_file(audio_file_path):
    
    audio, sample_rate = librosa.load(audio_file_path, sr=feature_extractor.sampling_rate)
    
   
    audio = audio[:feature_extractor.sampling_rate * 30]  
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True, do_normalize=True)
    
    # Move input features to the correct device
    input_features = inputs['input_features'].to(device)
    
    return input_features

# Function to get the classification result with confidence
def classify_audio_file(audio_file_path):
    input_features = preprocess_single_file(audio_file_path)
    
    with torch.no_grad():
        
        logits = model(input_features=input_features).logits
    
    # Apply softmax to logits to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get the predicted label and its confidence
    predicted_label = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, predicted_label].item() 

    
    label_map = {0: "AD", 1: "CN"}
    predicted_class = label_map[predicted_label]
    
    return predicted_class, confidence
