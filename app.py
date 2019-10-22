#!flask/bin/python
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas
from annoy import AnnoyIndex
from document_embedding import document_embedding

embedding_size = 300

song_table = pandas.read_hdf('songs.h5')
song_embeddings = AnnoyIndex(embedding_size, 'angular')
song_embeddings.load('test.ann')


app = Flask(__name__, static_url_path='')
# enable cross-origin
CORS(app)

@app.route('/', methods=['POST'])
def post():
    print(request.is_json)
    content = request.get_json()
    print(content['query'])
    q = document_embedding(content['query'])
    neighbours = song_embeddings.get_nns_by_vector(q, 5, search_k=-1, include_distances=True)
    response = []
    for index, n in enumerate(neighbours[0]):
        song_data = song_table.iloc[n].to_dict()
        song_data['distance'] = neighbours[1][index]
        response.append(song_data)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)