import torch
from scipy import spatial
import pandas as pd

def readModel (path):
    return torch.load(path ,map_location='cuda:0' if torch.cuda.is_available() else 'cpu')

#input is entity/relation as string and output is the corresponding vector
def getEntityEmbedHead (entity,model):
    index = model.ent2id[entity]
    return model.ent_h_embs.state_dict()['weight'][index]

def getEntityEmbedTail (entity,model):
    index = model.ent2id[entity]
    return model.ent_t_embs.state_dict()['weight'][index]

def getRelationEmbed (relation,model):
    index = model.rel2id[relation]
    return model.rel_embs.state_dict()['weight'][index]

#returns topN closest entities
def findClosestEntities (entity, model, topN):
    vecTail = getEntityEmbedTail(entity,model)
    vecHead = getEntityEmbedHead(entity,model)
    similarities = []
    for key, value in model.ent2id.items():
        distanceTail = 1 - spatial.distance.cosine(vecTail,model.ent_t_embs.state_dict()['weight'][value])
        distanceHead = 1 - spatial.distance.cosine(vecHead,model.ent_h_embs.state_dict()['weight'][value])
        meanSim = (distanceHead + distanceTail) / 2
        similarities.append([key,meanSim])
    print(sorted(similarities, key = lambda x: float(x[1]),reverse=True)[1:topN+1])

#similarity between entities
def getSim (entity1,entity2, head, model):
    if (head):
        vector1 = getEntityEmbedHead(entity1,model)
        vector2 = getEntityEmbedHead(entity2,model)
        distance = 1 - spatial.distance.cosine(vector1,vector2)
        return "Head Embedding: The cosine similarity between " + str(entity1) + " and " + str(entity2) + " is: " + str(distance) + "\n"
    else:
        vector1 = getEntityEmbedTail(entity1,model)
        vector2 = getEntityEmbedTail(entity2,model)
        distance = 1 - spatial.distance.cosine(vector1,vector2)
        return "Tail Embedding: The cosine similarity between " + str(entity1) + " and " + str(entity2) + " is: " + str(distance) + "\n"

def getMeanSim (entity1,entity2, model):
    vector1 = getEntityEmbedHead(entity1,model)
    vector2 = getEntityEmbedHead(entity2,model)
    distance1 = 1 - spatial.distance.cosine(vector1,vector2)
    vector3 = getEntityEmbedTail(entity1,model)
    vector4 = getEntityEmbedTail(entity2,model)
    distance2 = 1 - spatial.distance.cosine(vector3,vector4)
    distance = (distance1 + distance2) / 2
    return "Similarity '" + str(entity1) + "' and '" + str(entity2) + "' = " + str(distance)