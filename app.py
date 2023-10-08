import streamlit as st
import requests
import librosa
import tensorflow as tf
import numpy as np


def preprocessing(audio_file, mode):
    # we want to resample audio to 16 kHz
    sr_new = 16000  # 16kHz sample rate
    x, sr = librosa.load(audio_file, sr=sr_new)

    # padding sound
    # because duration of sound is dominantly 20 s and all of sample rate is 22050
    # we want to pad or truncated sound which is below or above 20 s respectively
    max_len = 5 * sr_new  # length of sound array = time x sample rate
    if x.shape[0] < max_len:
        # padding with zero
        pad_width = max_len - x.shape[0]
        x = np.pad(x, (0, pad_width))
    elif x.shape[0] > max_len:
        # truncated
        x = x[:max_len]

    if mode == 'mfcc':
        feature = librosa.feature.mfcc(y=x, sr=sr_new)

    elif mode == 'log_mel':
        feature = librosa.feature.melspectrogram(y=x, sr=sr_new, n_mels=128, fmax=8000)
        feature = librosa.power_to_db(feature, ref=np.max)

    return feature


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(
    page_title="Lung-Detect",
    page_icon="static/lung.svg",
    layout="wide",
)

st.title("Lung-Detect: Detecting human lung disease by voice analysis")
st.write("### Select an audio file in wav format for analysis")
st.write("#### Lung-Detect can detect Chronic Disease, Healthy, Non-Chronic Disease")


def main():
    dr = False
    st.write("---------------------------")
    voice_input = st.file_uploader("**Choose an audio file in wav format:** ", type=["wav"])
    if voice_input:
        st.write("------------------------------------------------------")
        dr = st.button("Analyze voice")

    if dr:
        model_path = 'saved_model/my_model'
        new_model = tf.keras.models.load_model(model_path)

        # preprocessing sound
        data = preprocessing(voice_input, mode='mfcc')
        data = np.array(data)
        print(data.shape)
        data = data.reshape((20, 157, 1))
        data = np.expand_dims(data, axis=0)

        datas = np.vstack([data])

        classes = new_model.predict(datas, batch_size=10)
        # classes = new_model.predict(datas)
        idx = np.argmax(classes)

        c_names = ['Chronic Disease', 'Healthy', 'Non-Chronic Disease']
        print('Result prediction: \n{}'.format(c_names[idx]))
        print('Confidence Percentage: {:.2f} %'.format(np.max(classes) * 100))

        st.write("------------------------------------------------------")
        st.write("detect success")
        st.write("Result prediction:", format(c_names[idx]))
        st.write("Confidence Percentage: {:.2f}".format(np.max(classes) * 100))
        st.write("-------------------------------")


if __name__ == '__main__':
    main()
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
