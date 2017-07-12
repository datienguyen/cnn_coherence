import numpy as np


def getString(docs=[], clusID=[], score=[], idx1=-1 , idx2=-1):
	s = docs[idx1] + ".M.100.T." + clusID[idx1] + ".html " + docs[idx2] + ".M.100.T." + clusID[idx2] + ".html "\
		+ str(score[idx1]) + "|" + str(score[idx2])

	return s

lines = [line.rstrip('\n') for line in open('duc03.train.score')]

docs = []
clusID = []
score = []

for line in lines:
	arr = line.split() 
	#print(arr)
	docs.append(arr[0])
	clusID.append(arr[1])
	score.append(float(arr[2]))

docID = list(set(docs))
pair_list = []


for id in docID:
	idxs = [i for i, x in enumerate(docs) if x == id ]
	#print(idxs)

	if score[idxs[4]] > score[idxs[3]]:
		pair_list.append(getString(docs, clusID, score, idxs[4], idxs[3]))

	if score[idxs[4]] > score[idxs[2]]:
		pair_list.append(getString(docs, clusID, score, idxs[4], idxs[2]))
		
	if score[idxs[4]] > score[idxs[1]]:
		pair_list.append(getString(docs, clusID, score, idxs[4], idxs[1]))
		
	if score[idxs[4]] > score[idxs[0]]:
		pair_list.append(getString(docs, clusID, score, idxs[4], idxs[0]))
		
	if score[idxs[3]] > score[idxs[2]]:
		pair_list.append(getString(docs, clusID, score, idxs[3], idxs[2]))
		
	if score[idxs[3]] > score[idxs[1]]:
		pair_list.append(getString(docs, clusID, score, idxs[3], idxs[1]))

	if score[idxs[3]] > score[idxs[0]]:
		pair_list.append(getString(docs, clusID, score, idxs[3], idxs[0]))
		
	if score[idxs[2]] > score[idxs[1]]:
		pair_list.append(getString(docs, clusID, score, idxs[2], idxs[1]))
		
	if score[idxs[2]] > score[idxs[0]]:
		pair_list.append(getString(docs, clusID, score, idxs[2], idxs[0]))
		
	if score[idxs[1]] > score[idxs[0]]:
		pair_list.append(getString(docs, clusID, score, idxs[1], idxs[0]))

print (pair_list)


with open ('duc03.pairs.train_dev', 'w') as fp:
	fp.write("\n".join(pair_list))













