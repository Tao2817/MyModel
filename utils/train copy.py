import torch, random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def metric(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / label
    mae = torch.mean(mae * mask)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
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
    # select random batch for making pictures
    rand_batch_idx = random.choice(range(total_len-1))
    all_batch_y = []
    for index, (x_batch, y_batch, time_stamp_batch, edge_index) in enumerate(loader):
        x_batch = x_batch.to(torch.float32).cuda()
        y_batch = y_batch.to(torch.float32).cuda()
        # x_batch_ = x_batch
        # x_batch = torch.cat((x_batch,y_batch),dim=1)
        edge_index = torch.from_numpy(edge_index).to(torch.int).cuda()
        time_stamp_batch = time_stamp_batch.to(torch.float32).squeeze(1).cuda()

        # #normalize
        # mean_batch = x_batch.mean(dim=1).unsqueeze(1)
        # std_batch = x_batch.std(dim=1).unsqueeze(1) + 1e-8
        # x_batch = (x_batch - mean_batch) / std_batch
        # y_batch = (y_batch - mean_batch) / std_batch

        # # test
        # x_ = torch.zeros_like(x_batch)
        # y_ = torch.zeros_like(y_batch)
        x_, y_  = model(x_batch,se,time_stamp_batch,edge_index)

        # #normalize
        # x_ = x_ * std_batch + mean_batch
        # y_ = y_ * std_batch + mean_batch
        # x_batch = x_batch * std_batch + mean_batch
        # y_batch = y_batch * std_batch + mean_batch
        # x_batch = x_batch_

        loss_x = mse_loss(x_batch.permute(0,3,2,1),x_.permute(0,3,2,1))
        # mae_x = my_model.loss(x_batch.permute(0,3,2,1),x_.permute(0,3,2,1))
        # loss_y = mse_loss(y_batch.permute(0,3,2,1),y_.permute(0,3,2,1))
        loss_y = mse_loss(y_batch.permute(0,3,2,1),y_.permute(0,3,2,1))


        # #x_5, y_5
        # loss = (5*loss_x + 5*loss_y)/10
        #x_1, y_9
        # loss = (1*loss_x + 9*loss_y)/10
        #x_0,y_10
        loss = loss_y + loss_x

        # mae_x = masked_mae_loss(x_batch.permute(0,3,2,1),x_.permute(0,3,2,1))
        # mae_y = masked_mae_loss(y_batch.permute(0,3,2,1),y_.permute(0,3,2,1))

        all_batch_y.append((y_batch.detach().cpu(),y_.detach().cpu()))
        if index == rand_batch_idx:
            rand_batch = x_batch, x_, y_batch, y_ 
        loss_batch[index] = loss

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if index % (total_len//10) == 0: 
    #         # sw.add_scalar(f"{mode}-loss",loss,epoch*total_len+index)
    #         # sw.add_scalar(f"{mode}-loss_avg",loss_batch[loss_batch!=0].mean(),epoch+index/total_len)
    #         # sw.add_scalars(f"{mode}-maes",{"mae_x":mae_x,"mae_y":mae_y},epoch+index/total_len)
    #         (mae_x, rmse_x, mape_x) = metric(x_batch, x_)
    #         (mae_y, rmse_y, mape_y) = metric(y_batch, y_)
            print(f"\tbatch: {index}/{total_len}\t\tloss: {loss}")
    if model.training : 
        scheduler.step()
    y_s = [i for (i,j) in all_batch_y]
    y_pred_s = [j for (i,j) in all_batch_y]
    y_s = torch.cat(y_s)
    y_pred_s = torch.cat(y_pred_s)
    (mae_y, rmse_y, mape_y) = metric(y_s, y_pred_s)
    print(f"\taverage loss: {loss_batch[loss_batch!=0].mean()}")
    print(f"\tmae: {mae_y}, rmse_y: {rmse_y} mape_y: {mape_y}")
    
    return rand_batch, all_batch_y, loss_batch[loss_batch!=0].mean(), (mae_y, rmse_y, mape_y)

def draw_pics(args, rand_batch, all_batch_y, mode, epoch):
    sample_num = 20
    x_batch, x_, y_batch, y_ = rand_batch
    sample_num = min(sample_num, args.batch_size, args.node_num)
    # assert sample_num<args.batch_size and sample_num<args.node_num
    # print picture regarding different time stamps(batch)
    time_index = random.sample(range(0,args.batch_size),k=sample_num)
    # all_batch_y =[_.cpu().detach() for _ in all_batch_y]
    x = np.array(x_batch.cpu().detach()[time_index,:,114,0])#batch=0, t=1-12, node_index=0
    x_pred = np.array(x_.cpu().detach()[time_index,:,114,0])
    y = np.array(y_batch.cpu().detach()[time_index,:,114,0])
    y_pred = np.array(y_.cpu().detach()[time_index,:,114,0])
    axis_x = np.array(range(args.his_len + args.fut_len))
    # plt.figure(figsize=(16,20),dpi=400)
    # for idx in range(sample_num):
    #     x_time = x[idx]
    #     x_pred_time = x_pred[idx]
    #     y_time = y[idx]
    #     y_pred_time = y_pred[idx]
    #     # print(x_time,x_pred_time,y_time,y_pred_time,sep="\n")
    #     plt.subplot(4, 5, idx+1)
    #     plt.xlim(-1,25)
    #     # plt.ylim(0,100)
    #     plt.title(f"time {time_index[idx]} ")
    #     plt.plot(axis_x[0:12],x_time,"r",label="x_true")
    #     plt.plot(axis_x[12:24],y_time,"g",label="y_true")
    #     plt.plot(axis_x[0:12],x_pred_time,"b",label="x_pred")
    #     plt.plot(axis_x[12:24],y_pred_time,"y",label="y_pred")
    #     plt.legend(loc="lower left")
    # plt.savefig(f"{args.save_path}/{mode}-epoch-{epoch}-time.jpg")
    # plt.clf()
    # # print picture regarding different nodes
    # x_batch = x_batch.cpu().detach()
    # node_num = x_batch.shape[2]
    # node_index = random.sample(range(0,node_num),k=sample_num)
    # x = np.array(x_batch.cpu().detach()[0,:,node_index,0].permute(1,0))#batch=0, t=1-12, node_index=0
    # x_pred = np.array(x_.cpu().detach()[0,:,node_index,0].permute(1,0))
    # y = np.array(y_batch.cpu().detach()[0,:,node_index,0].permute(1,0))
    # y_pred = np.array(y_.cpu().detach()[0,:,node_index,0].permute(1,0))
    # axis_x = np.array(range(12+12))
    # plt.figure(figsize=(16,20),dpi=400)
    # for idx in range(sample_num):
    #     x_node = x[idx]
    #     x_pred_node = x_pred[idx]
    #     y_node = y[idx]
    #     y_pred_node = y_pred[idx]
    #     # print(x_node,x_pred_node,y_node,y_pred_node,sep="\n")
    #     plt.subplot(4, 5, idx+1)
    #     plt.xlim(-1,25)
    #     # plt.ylim(0,100)
    #     plt.title(f"node {node_index[idx]} ")
    #     plt.plot(axis_x[0:12],x_node,"r",label="x_true")
    #     plt.plot(axis_x[12:24],y_node,"g",label="y_true")
    #     plt.plot(axis_x[0:12],x_pred_node,"b",label="x_pred")
    #     plt.plot(axis_x[12:24],y_pred_node,"y",label="y_pred")
    #     plt.legend(loc="lower left")
    # plt.savefig(f"{args.save_path}/{mode}-epoch-{epoch}-node.jpg")
    # plt.clf()

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