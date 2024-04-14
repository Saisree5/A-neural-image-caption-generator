from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import Img_form
from .models import Img_model
# Create your views here.
from django.shortcuts import redirect
import string
import numpy as np
from PIL import Image
import os
import pickle
from pickle import load
import keras
from nltk.translate.bleu_score import corpus_bleu
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import DenseNet201
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.translate.bleu_score import corpus_bleu
def hotel_image_view(request):
 
    if request.method == 'POST':
        form = Img_form(request.POST, request.FILES)
 
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = Img_form()
    return render(request, 'img.html', {'form': form})
 
 
def success(request):
    return render(request, 'success.html',{})

def display_hotel_images(request):
    
    if request.method == 'GET':
 
        # getting all the objects of hotel.
        img = Img_model.objects.all()
        return render(request, 'display.html',
                       {'Cap_img': img})
def recent(request):
    if request.method=='GET':
        img=Img_model.objects.all()
        return render(request,'recent.html',{'Cap_img':img})
from django.shortcuts import redirect

def cap2(request, redirect=True):
    try:
        id_val = request.POST["id_value"]
        p = Img_model.objects.get(id=id_val)
        v = p.img.url
        image_path = 'E:/gui/uploder' + v
        max_length = 32
        tokenizer = load(open("captionimg/models/tokenizer.p", "rb"))
        model = load_model('captionimg/models/xce_model.h5')
        xception_model = Xception()
        xception_model = keras.models.Model(inputs=xception_model.inputs, outputs=xception_model.layers[-2].output)

        # Display the image
        img = load_img(image_path)

        # Load and preprocess the image
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract features from the image using Xception
        feature = xception_model.predict(image, verbose=0)

        # Generate caption using the trained model
        caption = generate_desc(model, tokenizer, feature, max_length)
        
        if redirect:
            # If redirect is True, render the template
            return render(request, 'path.html', {'res': caption, "image": p})
        else:
            # If redirect is False, return the result only
            return {'res': caption, "image": p}
    except KeyError:
        return HttpResponse("ID value is missing.")
    except Img_model.DoesNotExist:
        return HttpResponse("Image not found.")

def cap22(request, redirect=True):
    try:
        id_val = request.POST["id_value"]
        p = Img_model.objects.get(id=id_val)
        v = p.img.url
        image_path = 'E:/gui/uploder' + v
        max_length = 32
        tokenizer = load(open("captionimg/models/tokenizer.p", "rb"))
        model = load_model('captionimg/models/modelGRU_9.h5')
        xception_model = Xception()
        xception_model = keras.models.Model(inputs=xception_model.inputs, outputs=xception_model.layers[-2].output)

        # Display the image
        img = load_img(image_path)

        # Load and preprocess the image
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract features from the image using Xception
        feature = xception_model.predict(image, verbose=0)

        # Generate caption using the trained model
        caption = generate_desc(model, tokenizer, feature, max_length)

        if redirect:
            # If redirect is True, render the template
            return render(request, 'path.html', {'res': caption, "image": p})
        else:
            # If redirect is False, return the result only
            return {'res': caption, "image": p}
    except KeyError:
        return HttpResponse("ID value is missing.")
    except Img_model.DoesNotExist:
        return HttpResponse("Image not found.")

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text
def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None

def cap1(request, redirect=True):
    try:
        id_val = request.POST["id_value"]
        p = Img_model.objects.get(id=id_val)
        v = p.img.url
        max_length = 34
        image_path = 'E:/gui/uploder' + v
        tokenizer = load(open("captionimg/models/tokenizer_vgg.pkl", "rb"))
        model = load_model('captionimg/models/LSTM_vgg.h5')
        vgg_model = VGG16()
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
        caption = generate_caption_for_image(model, image_path, vgg_model, tokenizer, max_length)

        if redirect:
            # If redirect is True, render the template
            return render(request, 'path.html', {'res': caption, "image": p})
        else:
            # If redirect is False, return the result only
            return {'res': caption, "image": p}
    except KeyError:
        return HttpResponse("ID value is missing.")
    except Img_model.DoesNotExist:
        return HttpResponse("Image not found.")
def cap11(request, redirect=True):
    try:
        id_val = request.POST["id_value"]
        p = Img_model.objects.get(id=id_val)
        v = p.img.url
        max_length = 34
        image_path = 'E:/gui/uploder' + v
        tokenizer = load(open("captionimg/models/tokenizer_vgg.pkl", "rb"))
        model = load_model('captionimg/models/GRU_vgg1.h5')
        vgg_model = VGG16()
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
        caption = generate_caption_for_image(model, image_path, vgg_model, tokenizer, max_length)

        if redirect:
            # If redirect is True, render the template
            return render(request, 'path.html', {'res': caption, "image": p})
        else:
            # If redirect is False, return the result only
            return {'res': caption, "image": p}
    except KeyError:
        return HttpResponse("ID value is missing.")
    except Img_model.DoesNotExist:
        return HttpResponse("Image not found.")

def generate_caption_for_image(model, image_path, vgg_model, tokenizer, max_length):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Extract features using VGG16
    feature = vgg_model.predict(image, verbose=0)

    # Generate caption for the image
    caption = predict_caption(model, feature, tokenizer, max_length)

    return caption

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # Add start tag for generation process
    in_text = 'startseq'

    # Iterate over the max length of sequence
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # Predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # Get index with high probability
        yhat = np.argmax(yhat)
        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        # Stop if word not found
        if word is None:
            break
        # Append word as input for generating next word
        in_text += " " + word
        # Stop if we reach end tag
        if word == 'endseq':
            break

    return in_text
class VGG16_Encoder(tf.keras.Model):
   # This encoder passes the features through a Fully connected layer
   def __init__(self, embedding_dim):
       super(VGG16_Encoder, self).__init__()
       # shape after fc == (batch_size, 49, embedding_dim)
       self.fc = tf.keras.layers.Dense(embedding_dim)
       self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

   def call(self, x):
       #x= self.dropout(x)
       x = self.fc(x)
       x = tf.nn.relu(x)
       return x



'''The encoder output(i.e. 'features'), hidden state(initialized to 0)(i.e. 'hidden') and
the decoder input (which is the start token)(i.e. 'x') is passed to the decoder.'''

class Rnn_Local_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Rnn_Local_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

        self.fc1 = tf.keras.layers.Dense(self.units)

        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
        self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

        self.fc2 = tf.keras.layers.Dense(vocab_size)

        # Implementing Attention Mechanism
        self.Uattn = tf.keras.layers.Dense(units)
        self.Wattn = tf.keras.layers.Dense(units)
        self.Vattn = tf.keras.layers.Dense(1)
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    def call(self, x, features, hidden):
        # features shape ==> (64,49,256) ==> Output from ENCODER
        # hidden shape == (batch_size, hidden_size) ==>(64,512)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size) ==> (64,1,512)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (64, 49, 1)
        # Attention Function
        '''e(ij) = f(s(t-1),h(j))'''
        ''' e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))'''

        score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))

        # self.Uattn(features) : (64,49,512)
        # self.Wattn(hidden_with_time_axis) : (64,1,512)
        # tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,49,512)
        # self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,49,1) ==> score

        # you get 1 at the last axis because you are applying score to self.Vattn
        # Then find Probability using Softmax
        '''attention_weights(alpha(ij)) = softmax(e(ij))'''

        attention_weights = tf.nn.softmax(score, axis=1)

        # attention_weights shape == (64, 49, 1)
        # Give weights to the different pixels in the image
        ''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) '''

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # Context Vector(64,256) = AttentionWeights(64,49,1) * features(64,49,256)
        # context_vector shape after sum == (64, 256)
        # x shape after passing through embedding == (64, 1, 256)

        x = self.embedding(x)
        # x shape after concatenation == (64, 1,  512)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU

        output, state = self.gru(x)
        # shape == (batch_size, max_length, hidden_size)

        x = self.fc1(output)
        # x shape == (batch_size * max_length, hidden_size)

        x = tf.reshape(x, (-1, x.shape[2]))

        # Adding Dropout and BatchNorm Layers
        x= self.dropout(x)
        x= self.batchnormalization(x)

        # output shape == (64 * 512)
        x = self.fc2(x)

        # shape : (64 * 8329(vocab))
        return x, state, attention_weights

    
def load_image(image_path):
   img = tf.io.read_file(image_path)
   img = tf.image.decode_jpeg(img, channels=3)
   img = tf.image.resize(img, (224, 224))
   img = preprocess_input(img)
   return img, image_path
def evaluate(image):
   #attention_plot = np.zeros((max_length, attention_features_shape))
   max_length = 33
   image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
   new_input = image_model.input
   hidden_layer = image_model.layers[-1].output
   image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
   from pickle import load
   tokenizer = load(open('captionimg/models/tokenizer_atten.p', 'rb'))
   embedding_dim = 256
   units = 512
   vocab_size = len(tokenizer.word_index) + 1
   encoder = VGG16_Encoder(embedding_dim)
   decoder = Rnn_Local_Decoder(embedding_dim, units, vocab_size)
   image_features_extract_model= tf.keras.models.load_model("captionimg/models/atten_model_vgg16-20240322T141948Z-001/atten_model_vgg16")
   
   hidden = decoder.reset_state(batch_size=1)
   temp_input = tf.expand_dims(load_image(image)[0], 0)
   img_tensor_val = image_features_extract_model(temp_input)
   img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
   encoder = tf.keras.models.load_model("captionimg/models/encoder_model_vgg16-20240322T142006Z-001/encoder_model_vgg16")
   decoder = tf.keras.models.load_model("captionimg/models/decoder_model_vgg16-20240322T141938Z-001/decoder_model_vgg16")
   
   features = encoder(img_tensor_val)
   dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
   result = []
   res=''
   for i in range(max_length):
       predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
       #attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
       predicted_id = tf.argmax(predictions[0]).numpy()
       result.append(tokenizer.index_word[predicted_id])

       if tokenizer.index_word[predicted_id] == '<end>':
           for i in result:
                res+=" "+i
           return res

       dec_input = tf.expand_dims([predicted_id], 0)
   #attention_plot = attention_plot[:len(result), :]
   
   for i in result:
       res+=" "+i

   return res

def attention_result(request, redirect=True):
    try:
        id_val = request.POST["id_value"]
        p = Img_model.objects.get(id=id_val)
        v = p.img.url
        image_path = 'E:/gui/uploder' + v
        result = evaluate(image_path)

        if redirect:
            # If redirect is True, render the template
            return render(request, 'path.html', {'res': result, "image": p})
        else:
            # If redirect is False, return the result only
            return {'res': result, "image": p}
    except KeyError:
        return HttpResponse("ID value is missing.")
    except Img_model.DoesNotExist:
        return HttpResponse("Image not found.")
def idx_to_word(integer,tokenizer):

    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None
def predict_caption_den(cap_model, image, tokenizer, max_length):
    model = DenseNet201()
    fe = Model(inputs=model.input, outputs=model.layers[-2].output)
    img_size = 224
    img = load_img(os.path.join('',image),target_size=(img_size,img_size))
    img = img_to_array(img)
    img = img/255.      
    img = np.expand_dims(img,axis=0)
    feature = fe.predict(img, verbose=0)
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = cap_model.predict([feature,sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text+= " " + word

        if word == 'endseq':
            break

    return in_text
def den_gru(request, redirect=True):
    try:
        id_val = request.POST["id_value"]
        p = Img_model.objects.get(id=id_val)
        v = p.img.url
        image_path = 'E:/gui/uploder/' + v
        max_length = 34
        tokenizer = load(open('captionimg/models/tokenizer_den.p', 'rb'))
        caption_model = tf.keras.models.load_model("captionimg/models/GRU_model_.h5")
        result = predict_caption_den(caption_model, image_path, tokenizer, max_length)

        if redirect:
            # If redirect is True, render the template
            return render(request, 'path.html', {'res': result, "image": p})
        else:
            # If redirect is False, return the result only
            return {'res': result, "image": p}
    except KeyError:
        return HttpResponse("ID value is missing.")
    except Img_model.DoesNotExist:
        return HttpResponse("Image not found.")
def den_lstm(request, redirect=True):
    try:
        id_val = request.POST["id_value"]
        p = Img_model.objects.get(id=id_val)
        v = p.img.url
        image_path = 'E:/gui/uploder/' + v
        max_length = 34
        tokenizer = load(open('captionimg/models/tokenizer_den.p', 'rb'))
        caption_model = tf.keras.models.load_model("captionimg/models/den_model.h5")
        result = predict_caption_den(caption_model, image_path, tokenizer, max_length)

        if redirect:
            # If redirect is True, render the template
            return render(request, 'path.html', {'res': result, "image": p})
        else:
            # If redirect is False, return the result only
            return {'res': result, "image": p}
    except KeyError:
        return HttpResponse("ID value is missing.")
    except Img_model.DoesNotExist:
        return HttpResponse("Image not found.")

class DenseNet201_Encoder(tf.keras.Model):
   # This encoder passes the features through a Fully connected layer
   def __init__(self, embedding_dim):
       super(DenseNet201_Encoder, self).__init__()
       # shape after fc == (batch_size, 49, embedding_dim)
       self.fc = tf.keras.layers.Dense(embedding_dim)
       self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

   def call(self, x):
       #x= self.dropout(x)
       x = self.fc(x)
       x = tf.nn.relu(x)
       return x

def evaluate_DEN(image):
   #attention_plot = np.zeros((max_length, attention_features_shape))
   max_length = 33
   image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
   new_input = image_model.input
   hidden_layer = image_model.layers[-1].output
   image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
   from pickle import load
   tokenizer = load(open('captionimg/models/tokenizer_atten.p', 'rb'))
   embedding_dim = 256
   units = 512
   vocab_size = len(tokenizer.word_index) + 1
   encoder = DenseNet201_Encoder(embedding_dim)
   decoder = Rnn_Local_Decoder(embedding_dim, units, vocab_size)
   image_features_extract_model=tf.keras.models.load_model("captionimg/models/atten_model_Den-20240317T171351Z-001/atten_model_Den")
   hidden = decoder.reset_state(batch_size=1)
   temp_input = tf.expand_dims(load_image(image)[0], 0)
   img_tensor_val = image_features_extract_model(temp_input)
   img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
   encoder = tf.keras.models.load_model("captionimg/models/encoder_model_Den-20240317T170035Z-001/encoder_model_Den")
   decoder = tf.keras.models.load_model("captionimg/models/decoder_model_Den-20240317T170035Z-001/decoder_model_Den")
   
   features = encoder(img_tensor_val)
   dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
   result = []
   res=''
   for i in range(max_length):
       predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
       #attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
       predicted_id = tf.argmax(predictions[0]).numpy()
       result.append(tokenizer.index_word[predicted_id])

       if tokenizer.index_word[predicted_id] == '<end>':
           for i in result:
                res+=" "+i
           return res

       dec_input = tf.expand_dims([predicted_id], 0)
   #attention_plot = attention_plot[:len(result), :]
   
   for i in result:
       res+=" "+i

   return res

def attention_den(request, redirect=True):
    try:
        id_val = request.POST["id_value"]
        p = Img_model.objects.get(id=id_val)
        v = p.img.url
        image_path = 'E:/gui/uploder' + v
        result = evaluate_DEN(image_path)

        if redirect:
            # If redirect is True, render the template
            return render(request, 'path.html', {'res': result, "image": p})
        else:
            # If redirect is False, return the result only
            return {'res': result, "image": p}
    except KeyError:
        return HttpResponse("ID value is missing.")
    except Img_model.DoesNotExist:
        return HttpResponse("Image not found.")
class Xception_Encoder(tf.keras.Model):
   # This encoder passes the features through a Fully connected layer
   def __init__(self, embedding_dim):
       super(Xception_Encoder, self).__init__()
       # shape after fc == (batch_size, 49, embedding_dim)
       self.fc = tf.keras.layers.Dense(embedding_dim)
       self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

   def call(self, x):
       #x= self.dropout(x)
       x = self.fc(x)
       x = tf.nn.relu(x)
       return x    
def evaluate_xcep(image):
   #attention_plot = np.zeros((max_length, attention_features_shape))
   max_length = 33
   image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
   new_input = image_model.input
   hidden_layer = image_model.layers[-1].output
   image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
   from pickle import load
   tokenizer = load(open('captionimg/models/tokenizer_atten.p', 'rb'))
   embedding_dim = 256
   units = 512
   vocab_size = len(tokenizer.word_index) + 1
   encoder = Xception_Encoder(embedding_dim)
   decoder = Rnn_Local_Decoder(embedding_dim, units, vocab_size)
   image_features_extract_model= tf.keras.models.load_model("captionimg/models/atten_model_Xcep-20240319T042159Z-001/atten_model_Xcep")
   
   hidden = decoder.reset_state(batch_size=1)
   temp_input = tf.expand_dims(load_image(image)[0], 0)
   img_tensor_val = image_features_extract_model(temp_input)
   img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
   encoder = tf.keras.models.load_model("captionimg/models/encoder_model_xce-20240319T042155Z-001/encoder_model_xce")
   decoder = tf.keras.models.load_model("captionimg/models/decoder_model_xce-20240319T042155Z-001/decoder_model_xce")
   
   features = encoder(img_tensor_val)
   dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
   result = []
   res=''
   for i in range(max_length):
       predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
       #attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
       predicted_id = tf.argmax(predictions[0]).numpy()
       result.append(tokenizer.index_word[predicted_id])

       if tokenizer.index_word[predicted_id] == '<end>':
           for i in result:
                res+=" "+i
           return res

       dec_input = tf.expand_dims([predicted_id], 0)
   #attention_plot = attention_plot[:len(result), :]
   
   for i in result:
       res+=" "+i

   return res

def attention_xcep(request, redirect=True):
    id_val = request.POST.get("id_value")
    if id_val:
        try:
            p = Img_model.objects.get(id=id_val)
            v = p.img.url
            image_path = 'E:/gui/uploder' + v
            result = evaluate_xcep(image_path)
            
            if redirect:
                # If redirect is True, render the template
                return render(request, 'path.html', {'res': result, "image": p })
            else:
                # If redirect is False, return the result only
                return {'res': result, "image": p }
        except Img_model.DoesNotExist:
            return HttpResponse("Image not found.")
    else:
        return HttpResponse("ID value is missing.")

    
def submission(request):
    if request.method == 'POST':
        print(request.POST)  # Print the POST data to the console
        selected_url = request.POST.get('action')
        # Redirect to the selected URL
        return redirect(selected_url)
    return HttpResponse("Function not found")

def analyze(request):
    
       
    res1 =cap2(request, redirect=False)
    res2 =cap22(request, redirect=False)
    res3 =cap1(request, redirect=False)
    res4 =cap11(request, redirect=False)
    res5 =attention_xcep(request, redirect=False)
    res6 =attention_result(request, redirect=False)
    res7 =attention_den(request, redirect=False)
    res8 =den_gru(request, redirect=False)
    res9 =den_lstm(request, redirect=False)
    if request.method == 'POST':
        
        id_value = request.POST.get('id_value')
        name='caption_'+id_value
        caption = request.POST.get(name)
        if caption=='':
            caption=''
            scores1=scores2=scores3=scores4=scores5=['-','-','-','-','-','-','-','-','-']
        else:
            print(id_value)
            
            # Split the generated caption into a list of words
            cap='start '+caption
            caption_list = cap.split(' ')
            print(cap)
            # Reference captions for each method
            #refs = [res1['res'].split(' '), res2['res'].split(' '), res3['res'].split(' '),
            #       res4['res'].split(' '), res5['res'].split(' '), res6['res'].split(' '),
            #      res7['res'].split(' '), res8['res'].split(' '), res9['res'].split(' ')]
            ref=[res1['res'], res2['res'], res3['res'],
                    res4['res'], res5['res'], res6['res'],
                    res7['res'], res8['res'], res9['res']]
            newref=[]
            for i in ref:
                if i.split(' ')[0]=='starseq':
                    newref.append(i.split(' ')[1:])
                elif i.split(' ')[0]=='start':
                    newref.append(i)
                else:
                    newref.append('start '+i)
            # Calculate BLEU scores for each method
            scores1 = [corpus_bleu([[word_tokenize(re)]],[word_tokenize(caption)]*len([word_tokenize(re)]), weights=(1.0, 0, 0, 0)) for re in newref]
            scores2 = [meteor_score( [word_tokenize(re)],word_tokenize(caption)) for re in newref]
            #rouge = Rouge()
            #scores3 = [rouge.get_scores(word_tokenize(re),word_tokenize(caption)) for re in newref]
            #cider = Cider()
            #scores4 = [cider.compute_score({0: [caption]},{i: [gen_caption] for i, gen_caption in enumerate(newref)})[0] ]        #spice = Spice()
            #scores5,_ = [spice.compute_score({0: [caption]}, {0: [re]}) for re in ref]     
    
    
    ana={}
    ana['Xception_LSTM']=res1['res']
    ana['Xception_GRU']=res2['res']
    ana['VGG16_LSTM']=res3['res']
    ana['VGG16_GRU']=res4['res']
    ana['Attention_GRU_Xcep']=res5['res']
    ana['Attention_GRU_VGG16']=res6['res']
    ana['Attention_GRU_DenseNet']=res7['res']
    ana['DenseNet_GRU']=res8['res']
    ana['DenseNet_LSTM']=res9['res']
    

    
    return render(request, 'analysis.html',
                       {"results":ana,"image":res1['image'],'caption': caption,'BLEU':scores1,
                        "meteor":scores2})
                     
    
    
    
    
    