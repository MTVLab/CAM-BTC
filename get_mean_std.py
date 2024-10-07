from tqdm import tqdm
import torch



def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3]) # [B,C,H,W] # 获取一个batch内每个通道的均值
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches #
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std