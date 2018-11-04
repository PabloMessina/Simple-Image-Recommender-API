import json
import sys
from enum import IntEnum
import numpy as np
from flask import Flask
from flask import json
from flask import request
from ModelHandlers import YoutubeLike2StagesModelHandler

class REC_IDS(IntEnum):
    YOUTUBE = 6

# ----- flask server definition --------

app = Flask(__name__)

@app.route('/rec_api')
def rec_api():
    print('---------- rec_api() -----------')

    # -- read request arguments
    rec_id = int(request.args['recommenderId'])
    rec_size = int(request.args['recSize'])
    print('rec_id = ', rec_id)
    print('rec_size = ', rec_size)

    # -- get recommendation
    if rec_id == REC_IDS.YOUTUBE:
        liked_ids = set([int(x) for x in request.args['likedArtworkIds'].split(",")])
        explained = request.args['explanation'] == 'true'
        print('liked_ids = ', liked_ids)
        print('explained = ', explained)
        recommendation = youtube_handler.get_recommendation(
            liked_ids, VALID_IDS, rec_size, explained)
    else: # unknown recommender
        return app.response_class(
            response=json.dumps({'message': 'unknown rec_id %d' % rec_id}),
            status=400,
            mimetype='application/json'                
        )

    # -- send recommendation
    return app.response_class(
        response=json.dumps(recommendation),
        status=200,
        mimetype='application/json'
    )

# ---------- main ------------------

# read config file
with open('config.json') as f:
    config = json.load(f)

# read command line arguments
develop = sys.argv[1] == 'develop'

# instance youtube-like model handler
if develop:
    youtube_handler = YoutubeLike2StagesModelHandler(
        config['develop_youtube_model_path'],
        config['develop_youtube_precomputed_tensors_path'],
        config['develop_resnet_precomputed_tensors_path'],
    )
else: # production
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    youtube_handler = YoutubeLike2StagesModelHandler(
        config['production_youtube_model_path'],
        config['production_youtube_precomputed_tensors_path'],
        config['production_resnet_precomputed_tensors_path'],
    )

# load valid ids
VALID_IDS = np.load('./VALID_IDS.npy')

# run app
app.run(threaded=True)