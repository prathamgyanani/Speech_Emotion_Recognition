from flask import Flask, render_template, request, jsonify
import soundfile
import librosa
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open("Emotion_Voice_Detection_Model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

def extract_feature(file_name, mfcc, chroma, mel):
    # Implementation of extract_feature function...
    with soundfile.SoundFile(file_name) as sound_file:
        if sound_file.channels > 1:
            mono_sound = sound_file.read(dtype="float32", always_2d=True)[:, 0]
        else:
            mono_sound = sound_file.read(dtype="float32")
        X = mono_sound
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    #print("mfccs shape:", mfccs.shape)
    #print("chroma shape:", chroma.shape)
    #print("mel shape:", mel.shape)
    return result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_blob' not in request.files:
        return jsonify({'error': 'No audio file part'})

    audio_blob = request.files['audio_blob']

    if audio_blob.filename == '':
        return jsonify({'error': 'No selected audio file'})

    try:
        # Save the uploaded audio file
        file_path = 'uploads' + secure_filename(audio_blob.filename)
        audio_blob.save(file_path)

        # Extract features from the saved audio file
        feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)

        # Make prediction using the loaded model
        prediction = model.predict([feature])[0]
        return render_template('after.html',data=prediction)
        #return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
