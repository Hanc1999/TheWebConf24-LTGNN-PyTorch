from datetime import datetime
import world
import utils
from world import cprint
import torch
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import time
import procedure as Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import networkx as nx
from torch_sparse import SparseTensor
from print_save import save_value

def record_in_an_epoch(F1_df, NDCG_df, test_results, epoch, path_excel,):
    ares = test_results[0] # here only considers the first result, which is a python dictionary
    f1 = ares['f1']
    ndcg = ares['ndcg']
    F1_df.loc[epoch + 1] = f1
    NDCG_df.loc[epoch + 1] = ndcg
    save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)

Recmodel = register.get_model_class(world.model_name)(world.config, dataset) # returns the model class
if not world.cpu_emb_table:
    Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

t = time.time()

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD: # by default do not
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

# make excel files to racord the training results
F1_df = pd.DataFrame(columns=world.topks)
NDCG_df = pd.DataFrame(columns=world.topks)
excel_dir = f"../experiment_results/{world.dataset}/{world.model_name}/"
today = datetime.today()
formatted_date = today.strftime('%Y%m%d')
excel_path = excel_dir + (
    f"{world.dataset}_{world.model_name}_{formatted_date}_{world.config['lr']}"
    f"_{world.config['decay']}_{world.config['lightGCN_n_layers']}"
    f"_{world.config['bpr_batch_size']}_{world.config['tune_index']}.xlsx"
)

try:
    # Training
    for epoch in range(world.TRAIN_epochs):
        if epoch % world.eval_interval == 0 and epoch != 0: # wierd logic
            cprint("[TEST]")
            if world.model_name in ['ltgnn']:
                test_results = Procedure.test_LTGNN(dataset, Recmodel, epoch, w, world.config['multicore'])
                record_in_an_epoch(F1_df, NDCG_df, test_results, epoch, excel_path,)
            else:
                if world.model_name == 'mf':
                    Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                else:
                    Procedure.Test_accelerated(dataset, Recmodel, epoch, w, world.config['multicore'])

        # Train one epoch
        start = time.time()
        if world.model_name in ['lgn-ns', 'lgn-vr', 'lgn-gas', 'ltgnn']:
            # train an epoch
            output_information = Procedure.train_LightGCN_NS(dataset, Recmodel, bpr.opt, epoch, neg_k=Neg_k,w=w)
        else:
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        t = time.time() - start
        
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information} time={t}')
        
        torch.save(Recmodel.state_dict(), weight_file)

finally:
    if world.tensorboard:
        w.close()
