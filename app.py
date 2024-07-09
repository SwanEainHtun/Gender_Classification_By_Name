
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/content/drive/MyDrive/Gender_Classification_by_Names/genderByName_model.h5')  # Replace with the actual path to your model

# Character index and parameters
char_index = {
  'a': 0,
 'h': 1,
 'g': 2,
 'l': 3,
 's': 4,
 'b': 5,
 'k': 6,
 'r': 7,
 'w': 8,
 'y': 9,
 'e': 10,
 'p': 11,
 'z': 12,
 't': 13,
 'u': 14,
 'j': 15,
 'o': 16,
 'n': 17,
 'c': 18,
 'd': 19,
 'EOW': 20,
 ' ': 21,
 'm': 22,
 'i': 23}
vocab_size = len(char_index) 
maxlen = 25
def preprocess_single_name(name, char_index, vocab_size, maxlen):
    name = name.lower()
    def set_encode(i):
        temp = np.zeros(vocab_size)
        temp[i] = 1
        return list(temp)

    new_list = []
    train_name = str(name)[0:maxlen]

    tmp = [set_encode(char_index[j]) for j in train_name]
    for k in range(0, maxlen - len(train_name)):
        tmp.append(set_encode(char_index["EOW"]))

    new_list.append(tmp)
    return np.array(new_list)

def predict_gender(name):
    # Preprocess the name
    processed_name = preprocess_single_name(name, char_index, vocab_size, maxlen)
    # Make the prediction
    prediction = model.predict(processed_name)
    # Convert prediction to label
    predicted_label = np.argmax(prediction, axis=1)[0]
    return 'Male 🧑' if predicted_label == 0 else 'Female 👱‍♀️'

# Streamlit app
st.title('Gender Prediction from Name')
name = st.text_input('Enter a name:', '')

if st.button('Predict Gender'):
    if name.strip():
        gender = predict_gender(name)
        st.write(f'The predicted gender for the name "{name}" is {gender}.')
    else:
        st.write("Please enter a name.")


