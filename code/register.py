import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['yelp2018', 'amazon-book', 'mba', 'instacart', 'instacart_full']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset in ['alibaba-ifashion', 'amazon3']:
    dataset = dataloader.LargeScaleLoader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
        
print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    # 'mf': model.PureMF,
    # 'lgn': model.LightGCN,
    # 'lgn-ns': model.NSLightGCN,
    # 'lgn-vr': model.VRLightGCN,
    # 'lgn-gas': model.GASLightGCN,
    'ltgnn': model.LTGNN,
}

def get_model_class(name):
    return MODELS[name]