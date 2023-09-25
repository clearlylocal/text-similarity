from sentence_transformers import SentenceTransformer, util
import time
import json
import re

model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')

data = json.loads(open('./input.json', 'r').read())

source = data['source']
targets = [re.sub(r'^\d+\. |[\[\]【】]', '', x) for x in data['targets']]

start_time = time.time()

source_embeddings = model.encode([source], convert_to_tensor=True)
target_embeddings = model.encode(targets, convert_to_tensor=True)

cosine_scores = util.cos_sim(source_embeddings, target_embeddings)

for i in range(len(targets)):
	print("{} \t\t {} \t\t Score: {:.4f}".format(source, targets[i], cosine_scores[0][i]))

print("--- %s seconds ---" % (time.time() - start_time))
