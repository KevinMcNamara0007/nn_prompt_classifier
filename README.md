# Load the model from the file
from tensorflow.keras.models import load_model
model2 = load_model('model.h5')

# check the model
model2.summary()

# create a sample sentence to test
sentence = "What is 2 + 2?"

# Create an inference
# tokenize and pad the input sentence
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentence)

# Append tokenized sequences to the X_train dataframe as a new column
sentence = tokenizer.texts_to_sequences([sentence])
max_sentence_length = 100

padded_sequence = pad_sequences(sentence, maxlen=max_sentence_length, padding='post')


padded_sequence_array = np.array(padded_sequence)

# create prediction
prediction = model2.predict(padded_sequence_array)

# print the prediction
print(prediction)
