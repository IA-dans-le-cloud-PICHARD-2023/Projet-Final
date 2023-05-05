from flask import Flask, render_template, request, session
import csv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv
import pickle
import pandas as pd
import xgboost as xgb
import time
from flask_socketio import SocketIO, emit
from queue import Queue
import random

client_id = "7a51c1f0b5c040a3873a4dec87f4d38c"
client_secret = "fc49bd0777984976a40a2ee49a532b1c"

def RecupData(data):
    # Renseignez le nom de la piste que vous souhaitez rechercher ci-dessous
    track_name = data

    # Initialisez l'authentification avec les identifiants de client Spotify
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Recherchez la piste
    results = sp.search(q=track_name, type='track')
    items = results['tracks']['items']

    # Si des pistes sont trouvées, affichez les informations demandées pour la première piste trouvée   
    if items:
        track = items[0]
        track_info = sp.audio_features(track['id'])[0]
        track_name = track['name']
        track_artist = track['artists'][0]['name']
        track_uri = track['uri']
        danceability = track_info['danceability']
        energy = track_info['energy']
        key = track_info['key']
        loudness = track_info['loudness']
        mode = track_info['mode']
        speechiness = track_info['speechiness']
        acousticness = track_info['acousticness']
        instrumentalness = track_info['instrumentalness']
        liveness = track_info['liveness']
        valence = track_info['valence']
        tempo = track_info['tempo']
        duration_ms = track_info['duration_ms']
        time_signature = track_info['time_signature']
        # chorus moyen
        chorus_hit = 40.10604
        audio_analysis = sp.audio_analysis(track['uri'])
        sections = len(audio_analysis['sections'])

        # Enregistrez les informations dans un fichier CSV
        with open('./outputs/datas/mytracks.csv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['track', 'artist', 'uri', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'chorus_hit', 'sections', 'target'])
            writer.writerow([track_name, track_artist, track_uri, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature, chorus_hit, sections])
        
    else:
        print("Aucune piste trouvée pour le nom :", track_name)


def PredictData():
    input_file = './outputs/models/final_model_xgb.bin'

    with open(input_file, 'rb') as f_in: 
        dv, model = pickle.load(f_in)
        
    df = pd.read_csv('./outputs/mytracks.csv')
    df = df.drop(['track', 'artist', 'uri', 'target'], axis=1)
    df = df.drop(['duration_ms'], axis=1)

    song = df.to_dict(orient='records')[0]

    X = dv.transform([song])
    dX = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())

    y_pred = model.predict(dX)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

queue = Queue()
queue = []

@app.route('/', methods=['GET'])
def index():
    session['position'] = len(queue) + 1
    return render_template('indexv2.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['musique']
    session['id'] = random.randint(1, 1000000)
    RecupData(data)
    time.sleep(5)
    y_pred = PredictData()
    print("y_pred : ", y_pred)
    position = session.get('position')
    queue.append((session['id'], position))
    socketio.emit('position', {'position': position}, room=session['id'])
    socketio.emit('prediction', {'prediction': y_pred}, room=session['id'])
    session['position'] = len(queue) + 1
    return render_template('indexv2.html', predict=y_pred)


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    for item in queue:
        if item[0] == request.sid:
            queue.remove(item)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
