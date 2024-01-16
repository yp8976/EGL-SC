import ml
import gowalla
import utils

import torch
import numpy as np
import time
import Procedure
from os.path import join
config = {}
def run(script_name):
    if script_name == "Movielens1M":
        config =  ml.config
    if script_name == "Movielens10M":
        config =  ml.config
    elif script_name == "Gowalla":
        config =  gowalla.config
    elif script_name == "Yelp2018":
        config =  ml.config
    elif script_name == "1M":
        config =  ml.config
    elif script_name == "Movielens100K":
        config =  ml.config 
    elif script_name == "AmazonElectronics":
        config =  ml.config 
    elif script_name == "AmazonCD":
        config =  ml.config 
    elif script_name == "DBLP":
        config =  ml.config 
    elif script_name == "huagong":
        config =  ml.config 
    elif script_name == "100K":
        config =  ml.config 
    elif script_name == "wiki":
        config =  ml.config 
    elif script_name == "Amazon-Book":
        config =  ml.config 
    else:
        print("ERROR None")
    import world
    from world import cprint
    world.dataset = config['dataset'] # "ml-1m"
    world.config['test_u_batch_size'] = config['testbatch']    # 100
    world.topks = [config['topk']]    # [20]
    world.device = config['device']
    world.config['anneal_cap'] = config['anneal_cap']
    world.config['total_anneal_steps'] = config['total_anneal_steps']
    import register
    from register import dataset

    def eval(user_emb,item_emb):

        utils.set_seed(world.seed)
        Recmodel = register.MODELS[world.model_name](world.config, dataset)
        Recmodel = Recmodel.to(world.device)
        Recmodel.embedding_user.weight = user_emb
        Recmodel.embedding_item.weight = item_emb

        w = None
        result = Procedure.Test(dataset, Recmodel, 0, w, world.config['multicore'])
        return result
    return eval

inner = run
