import coreConfig as cc
import dataset as usd
import MyMetrics as met
from MyModels import * 
exec(cc.stmts)

def test_single_epoch(model , data_loader , loss_fun , device ) :
    model.eval()
    with torch.inference_mode() :
        for inp , tar in data_loader :
            inp , tar = inp.to(device) , tar.to(device)
            logits = model(inp.view(-1, 64, 171)) 
            
            loss = loss_fun(logits , tar)
            acc = met.acc(logits , tar)
            
        print(f"testing accuracy {acc}")

def train_single_epoch(model , data_loader , loss_fun , optimiser , device ) :
    model.train()
    for inp , tar in data_loader :
        inp , tar = inp.to(device) , tar.to(device)
        
        logits = model(inp.view(-1, 64, 171))
        loss = loss_fun(logits , tar)
        acc = met.acc(logits , tar)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"\ntraining accuracy {acc}")

def fit(model , trLdr , tstLdr  , loss_fun , optimiser , device , epochs):

    for i in tqdm(range(epochs) , desc = "->"):
        train_single_epoch(model , trLdr , loss_fun , optimiser , device )
        test_single_epoch(model , tstLdr , loss_fun , device)
    

if __name__ == "__main__":
    #idea to perform kfold 
    testFoldSet = [[j+1 for j in i] for i in it.permutations([i for i in range(cc.kfold)], r=cc.num_test_folds)]

    Spec = cc.models[cc.currModel]["spec"]
    ToDB = cc.models[cc.currModel]["toDB"]
    params = cc.models[cc.currModel]["params"]
    pt_file = cc.models[cc.currModel]["path"]
    print(pt_file)
    
    print(testFoldSet)
    loaders = [(DataLoader(usd.UrbanSoundDataset(
                    spec = Spec,
                    toDB = ToDB ,
                    train = True ,
                    test_fold = fold
                    ),
                batch_size=cc.batch_size, shuffle=True),
                DataLoader(usd.UrbanSoundDataset(
                    spec = Spec,
                    toDB = ToDB ,
                    train = False ,
                    test_fold = fold
                    ),
                batch_size=cc.batch_size, shuffle=True))
               for fold in testFoldSet]
    model = globals()[cc.currModel](*params)

    if os.path.exists(pt_file):
        status = model.load_state_dict(torch.load(pt_file))
        print(status)
        
    model = model.to(device)
    loss_fun = eval(cc.models[cc.currModel]["loss_fun"])
    optimizer = eval(cc.models[cc.currModel]["optimizer"])

    print(loss_fun , optimizer)
    

    print("PERFORMING K-FOLDS")
    for trLdr , tstLdr in loaders :
        fit(model , trLdr , tstLdr  , loss_fun , optimizer , device , cc.epochs)
    print("training finished")

    
