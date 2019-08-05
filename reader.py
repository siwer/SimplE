import torch

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