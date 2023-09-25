from flask import Flask, request, Response
from flask_cors import CORS
import logging
from sentence_transformers import SentenceTransformer, util
import time
import json

model_name = 'sentence-transformers/stsb-xlm-r-multilingual'
model = SentenceTransformer(model_name)

min_log_level = logging.ERROR

# logger used by flask
logger = logging.getLogger('werkzeug')
logger.setLevel(min_log_level)

def print(data):
	logger.log(min_log_level, data)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
	return Response(response='ok', status=200)

@app.route(f'/models/{model_name}', methods = ['POST'])
def lemmatize():
	body = request.get_json()

	inputs = body['inputs']
	source = inputs['source_sentence']
	targets = inputs['sentences']

	start_time = time.time()

	source_embeddings = model.encode([source], convert_to_tensor=True)
	target_embeddings = model.encode(targets, convert_to_tensor=True)

	cosine_scores = util.cos_sim(source_embeddings, target_embeddings)

	print(f'--- {time.time() - start_time} seconds ---')

	return Response(
		response=json.dumps(cosine_scores[0].tolist()),
		status=200,
		mimetype='application/json',
	)

if __name__ == '__main__':
	app.run()
