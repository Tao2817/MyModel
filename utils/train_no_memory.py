import torch, random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def metric(pred, label):
    # 缺失数据（速度为0）不加入loss计算
    # pred = torch.where(pred==0, torch.nan, pred)
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.nanmean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / label
    mae = torch.nanmean(mae * mask)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.nanmean(rmse))
    mape = mape * mask
    mape = torch.nanmean(mape)
    return mae, rmse, mape

def mae_loss(x_true, x_pred):
    mae = torch.abs(torch.sub(x_pred, x_true)).type(torch.float32)
    mae = torch.nanmean(mae)
    return mae

def mse_loss(x_true, x_pred):
    mae = torch.abs(torch.sub(x_pred, x_true)).type(torch.float32)
    rmse = mae ** 2
    rmse = torch.sqrt(torch.nanmean(rmse))
    return rmse

    
def masked_mae_loss(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def train(mode, loader, model, se, optimizer=None, scheduler=None):
    total_len = len(loader)
    loss_batch = torch.zeros((total_len,1))
    all_batch_y = []
    for index, (x_batch, y_batch, time_stamp_batch, edge_index) in enumerate(loader):
        x_batch = x_batch.to(torch.float32).cuda()
        y_batch = y_batch.to(torch.float32).cuda()

        x_batch = loader.dataset.norm(x_batch)

        edge_index = torch.from_numpy(edge_index).to(torch.int64).cuda()
        time_stamp_batch = time_stamp_batch.to(torch.float32).squeeze(1).cuda()


        y_  = model(x_batch,se,time_stamp_batch,edge_index)

        y_ = loader.dataset.norm_inv(y_)

        y_ = torch.where(y_batch==0, 0, y_)


        loss_y = mae_loss(y_batch.permute(0,3,2,1),y_.permute(0,3,2,1))


        loss = loss_y 
        # loss = loss_y

        all_batch_y.append((y_batch.detach().cpu(),y_.detach().cpu()))
        loss_batch[index] = loss

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if index % (total_len//10) == 0: 
            print(f"\tbatch: {index}/{total_len}\t\tloss: {loss}")
    if model.training : 
        scheduler.step()
    y_s = [i for (i,j) in all_batch_y]
    y_pred_s = [j for (i,j) in all_batch_y]
    y_s = torch.cat(y_s)
    y_pred_s = torch.cat(y_pred_s)
    (mae_y, rmse_y, mape_y) = metric(y_s, y_pred_s)
    print(f"\taverage loss: {loss_batch[loss_batch!=0].nanmean()}")
    print(f"\tmae: {mae_y}, rmse_y: {rmse_y} mape_y: {mape_y}")
    
    return all_batch_y, loss_batch[loss_batch!=0].nanmean(), (mae_y, rmse_y, mape_y)

def draw_pics(args, all_batch_y, mode, epoch):
    sample_num = 20
    sample_num = min(sample_num, args.batch_size, args.node_num)
    time_index = random.sample(range(0,args.batch_size),k=sample_num)
    axis_x = np.array(range(args.his_len + args.fut_len))

    #concat the preds
    all_y_real, all_y_pred = [_[0] for _ in all_batch_y], [_[1] for _ in all_batch_y]
    all_y_real = all_y_real[0:-1]
    all_y_pred = all_y_pred[0:-1]
    all_y_real = torch.cat(all_y_real,dim=0)
    all_y_pred = torch.cat(all_y_pred,dim=0)
    index = torch.tensor([i for i in range(all_y_pred.shape[0]) if i % args.fut_len == 0])
    all_y_real = all_y_real.index_select(0,index)
    all_y_pred = all_y_pred.index_select(0,index)
    time_index, pred_len, node_index, _ = all_y_real.shape
    all_y_real = all_y_real.reshape(time_index * pred_len, node_index, -1)
    all_y_pred = all_y_pred.reshape(time_index * pred_len, node_index, -1)
    all_y_real = all_y_real.split(288)
    all_y_pred = all_y_pred.split(288)

    days = len(all_y_real)
    rand_idx = random.choices(range(days),k=min(days,10))

    all_y_real = [all_y_real[i] for i in rand_idx]
    all_y_pred = [all_y_pred[i] for i in rand_idx]
    days = len(all_y_real)

    fig, axs = plt.subplots(days, 1, figsize=(20,days*8), dpi=200)
    for idx, (y_real,y_pred) in enumerate(zip(all_y_real,all_y_pred)):
        rand_node = random.sample(range(0,y_real.shape[1]),k=1)
        if isinstance(axs,np.ndarray):
            ax = axs[idx]
        else:
            ax = axs
        axis_x = list(range(y_pred.shape[0]))
        ax.set_title(f"day {idx}, node {rand_node}:")
        ax.plot(axis_x,y_real[:,rand_node].squeeze(-1).numpy(),"r",label="y_ture")
        ax.plot(axis_x,y_pred[:,rand_node].squeeze(-1).numpy(),"b",label="y_pred")
        ax.legend(loc="lower left")
        ax.set_ylim(0,100)
        ax.set_xticks(np.arange(0,axis_x[-1],args.his_len))
        ax.grid(True, axis="x", linestyle='-.')
    fig.savefig(f"{args.save_path}/{mode}-epoch-{epoch}-rand_node-rand10_day.jpg")
    plt.close("all")