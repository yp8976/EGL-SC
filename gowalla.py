import numpy as np
import os
import torch

from torch import nn
import math
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
import numpy as np
from datetime import datetime
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm
import torch.multiprocessing as mp
from multiprocessing import Pool
from scipy.stats import entropy, wasserstein_distance
import multiprocessing
from functools import partial
import scipy.sparse as sp
import pickle
import os
import sys
import time


def read_dataset(filename):
    orgin_data = []
    with open(filename, encoding="UTF-8") as fin:
        line = fin.readline()
        while line:
            user, item, rating = line.strip().split("\t")
            orgin_data.append((int(user)-1, int(item)-1, float(rating)))
            line = fin.readline()

    user, item = set(), set()
    for u, v, r in orgin_data:
        user.add(u)
        item.add(v)
    user_list = list(user)
    item_list = list(item)
    uLen = max(user_list)+1
    vLen = max(item_list)+1

    return orgin_data, user_list, item_list, uLen, vLen


def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1
    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])

    return train_data, test_data, n_user, m_item


def create_RMN(train_data, uLen, vLen):
    data = []
    W = torch.zeros((uLen, vLen))
    for w in tqdm(train_data):
        u, v = w
        data.append((u, v))
        W[u][v] = 1

    return W


def sumpow(x, k):
    sum = 0
    for i in range(k+1):
        sum += math.pow(x, i)
    return sum


def lightgcn_init(user_path, item_path):
    print("初始化向量")
    if os.access(user_path, os.F_OK) and os.access(item_path, os.F_OK):
        print("正在加载初始向量")
        u_vector = torch.load(user_path)
        v_vector = torch.load(item_path)
        u_vectors = u_vector.cpu().detach()
        u_vectors = torch.nn.Parameter(u_vectors)
        v_vectors = v_vector.cpu().detach()
        v_vectors = torch.nn.Parameter(v_vectors)
    else:
        raise Exception("没有找到初始向量")
    return u_vectors, v_vectors


def utils2(x, y, p=2):
    wasserstein_distance = torch.abs(
        (
            torch.sort(x.transpose(0, 1), dim=1)[0]
            - torch.sort(y.transpose(0, 1), dim=1)[0]
        )
    )
    wasserstein_distance = torch.pow(
        torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    wasserstein_distance = torch.pow(
        torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance


def init_vectors(rank, uLen, vLen):
    print("初始化向量")
    u_vectors = torch.empty(uLen, rank)
    u_vectors = torch.nn.init.kaiming_normal_(u_vectors)
    u_vectors = preprocessing.normalize(u_vectors, norm='l2')
    v_vectors = torch.empty(vLen, rank)
    v_vectors = torch.nn.init.kaiming_normal_(v_vectors)
    v_vectors = preprocessing.normalize(v_vectors, norm='l2')

    return u_vectors, v_vectors


def gpu():
    from pynvml import (nvmlInit,
                        nvmlDeviceGetCount,
                        nvmlDeviceGetHandleByIndex,
                        nvmlDeviceGetMemoryInfo,
                        nvmlShutdown)

    nvmlInit()

    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):

        handle = nvmlDeviceGetHandleByIndex(i)

        info = nvmlDeviceGetMemoryInfo(handle)

        print(f"Device {i}: ")
        print(f"Total memory: {info.total/1024**2} MB")
        print(f"Free memory: {info.free/1024**2} MB")
        print(f"Used memory: {info.used/1024**2} MB")
    nvmlShutdown()


class GRU_Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        '''
        :param in_dim: 输入向量的维度
        :param hidden_dim: 输出的隐藏层维度
        '''
        super(GRU_Cell, self).__init__()
        self.rx_linear = nn.Linear(in_dim, hidden_dim)
        self.rh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.zx_linear = nn.Linear(in_dim, hidden_dim)
        self.zh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.hx_linear = nn.Linear(in_dim, hidden_dim)
        self.hh_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h_1):
        r = torch.sigmoid(self.rx_linear(x)+self.rh_linear(h_1))
        z = torch.sigmoid(self.zx_linear(x)+self.zh_linear(h_1))
        h_ = torch.tanh(self.hx_linear(x)+self.hh_linear(r*h_1))
        h = z*h_1+(1-z)*h_
        return h


class Net(nn.Module):
    def __init__(self, config, u_vectors, v_vectors, q_dims=None, dropout=0.5):
        super(Net, self).__init__()
        u_len = config['ulen']
        v_len = config['vlen']
        rank = config['rank']
        self.u = nn.Embedding(u_len, rank)
        self.u.weight = u_vectors
        self.v = nn.Embedding(v_len, rank)
        self.v.weight = v_vectors

        p_dims = config["p_dims"]
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1]]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.update = 0
        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.drop = nn.Dropout(dropout)

        self.mlplist = nn.ModuleList(
            [nn.Sequential(nn.Linear(rank, rank*2), nn.ReLU(), nn.Linear(2*rank, rank))])
        self.vaelist = nn.ModuleList(
            [nn.Sequential(nn.Linear(v_len, 600), nn.Linear(600, 400))])

        self.GRU = nn.GRU(input_size=1, hidden_size=200, batch_first=True)
        self.rnn_cell = GRU_Cell(in_dim=1, hidden_dim=400)
        self.init_weights()

    def gru(self, x, h=None):
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_dim)

        outs = []

        for t in range(x.shape[1]):
            seq_x = x[:, t, :]
            h = self.rnn_cell(seq_x, h)
            outs.append(torch.unsqueeze(h, 1))

        outs = torch.cat(outs, dim=1)
        return outs, h

    def nfm(self):
        interaction = torch.mm(self.u.weight, self.v.weight.t())
        square_of_sum = torch.pow(interaction, 2)
        self.fc_layers = nn.ModuleList()
        hidden_dims = [600, 400, 200]
        for idx, (in_size, out_size) in enumerate(zip([vLen] + hidden_dims[:-1], hidden_dims)):
            self.fc_layers.append(nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())
        inter_term = 0.5 * (square_of_sum)
        for layer in self.fc_layers.to(device):
            inter_term = layer(inter_term)
        output = inter_term

        return output

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                W = self.nfm()
                W = W.unsqueeze(0)
                h = h.unsqueeze(-1)
                output, h = self.GRU(h, W)
                print(output.shape, h.shape)
                h1 = h.squeeze(0)
                h2 = output.sum(dim=1)
                print(h1.shape, h2.shape)
                mu = h1
                logvar = h2

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            torch.nn.init.kaiming_normal_(
                layer.weight.data, nonlinearity='tanh')
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            torch.nn.init.kaiming_normal_(
                layer.weight.data, nonlinearity='tanh')
            layer.bias.data.normal_(0.0, 0.001)

    def vae(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 *
                         self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        kl_loss = (
            -0.5
            * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        ce_loss = -(F.log_softmax(z, 1) * input).sum(1).mean()

        return z, kl_loss+ce_loss

    def forward(self, W):
        RR = torch.mm(self.u.weight, self.v.weight.t())
        torch.cuda.empty_cache()
        logp_R = F.log_softmax(RR, dim=-1)
        tenW = W.to(device)
        p_R = F.softmax(tenW, dim=-1)
        del tenW
        torch.cuda.empty_cache()
        kl_sum_R = torch.nn.KLDivLoss(reduction='sum')(logp_R, p_R)
        del logp_R, p_R
        torch.cuda.empty_cache()
        C = utils2(W.to(device), RR)
        return RR, kl_sum_R+0.0005*C


def train_gnn(config, model, optimizer, W, W2):
    best = 0.0
    bestR = []
    bestE = 0
    converged_epochs = 0
    try:
        should_stop = False
        for epoch in range(config['epochs']):
            start_time = time.time()
            model.train()
            optimizer.zero_grad()
            R, loss = model(W)
            loss.backward()
            optimizer.step()
            model.eval()
            val = eval(model.u.weight, model.v.weight)
            print("|Epoch", epoch, "|Val", val)
            val = val['ndcg']
            converged_epochs += 1
            if val > best:
                best = val
                bestE = epoch
                bestR = R
                converged_epochs = 0
            if converged_epochs >= 10 and epoch > 100:
                print('模型收敛，停止训练。最优ndcg值为：', best,
                      "最优epoch为：\n", bestE, "最优R为：\n", bestR)
                print("保存模型参数")
                break
            if epoch == config['epochs'] - 1:
                print('模型收敛，停止训练。最优ndcg值为：', best, "最优R为：\n", bestR)
            if should_stop:
                break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print('=' * 50)
    train_time = time.strftime(
        "%H: %M: %S", time.gmtime(time.time() - start_time))
    print("训练时间:", train_time)
    print('| 训练结束 | 最优轮:{0}|最优 NDCG@{1}:{2}'.format(bestE,
          config['topk'], best))
    print('=' * 50)

    return bestR


def train_vae(config, model, optimizer, W, W2):
    best = 0.0
    bestR = []
    bestE = 0
    converged_epochs = 0
    try:
        should_stop = False
        for epoch in range(config['epochs']):
            start_time = time.time()
            model.train()
            optimizer.zero_grad()
            R, loss = model.vae(W)
            loss.backward()
            optimizer.step()
            model.eval()
            val = multivae.eval_ndcg(model, W2, R, k=20)
            val_r = multivae.eval_recall(model, W2, R, k=20)
            percison = multivae.eval_precision(model, W2, R, k=20)
            print('=' * 89)
            print('| Epoch {:2d}|loss {:4.5f} | percision {:4.5f} | r20 {:4.5f}| n20 {:4.5f} |'.format(
                epoch, loss, percison, val_r, val))
            converged_epochs += 1
            if val_r > best:
                best = val_r
                bestE = epoch
                bestR = R
                converged_epochs = 0
            if converged_epochs >= 10 and epoch > 100:
                print('模型收敛，停止训练。最优ndcg值为：', best,
                      "最优epoch为：\n", bestE, "最优R为：\n", bestR)
                print("保存模型参数")
                break
            if epoch == config['epochs'] - 1:
                print('模型收敛，停止训练。最优ndcg值为：', best, "最优R为：\n", bestR)
            if should_stop:
                break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print('=' * 50)
    train_time = time.strftime(
        "%H: %M: %S", time.gmtime(time.time() - start_time))
    print("训练时间:", train_time)
    print('| 训练结束 | 最优轮:{0}|最优 NDCG@{1}:{2}'.format(bestE,
          config['topk'], best))
    print('=' * 50)

    return bestR


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {
    'dataset': 'Yelp2018',
    'topk': 20,
    'lr': 1e-3,
    'wd': 0.0,
    'rank': 128,
    'batch_size': 512,
    'testbatch': 100,
    'epochs': 1000,
    'total_anneal_steps': 200000,
    'anneal_cap': 0.2,
    'seed': 2020,
    'device': device
}

if __name__ == "__main__":
    from evalu import run  # 导入评估
    import multivae
    import os

    print('Random seed: {}'.format(config['seed']))
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    eval = run(config['dataset'])
    rank = 128
    epochs = config['epochs']
    lr = config['lr']
    iters = 1
    top_n = config['topk']
    dataset = config['dataset']
    batch_size = config['batch_size']
    test_batch_size = config['testbatch']

    print("dataset", dataset, "rank:", rank, "epochs:",
          epochs, "lr:", lr, "topK@", top_n, "iters:", iters)
    path = os.path.dirname(os.path.dirname(__file__))
    train_file = r"../data/"+dataset+"/train.txt"
    test_file = r"../data/"+dataset+"/test.txt"
    print("train_file:", train_file)
    print("test_file:", test_file)
    train_data, test_data, uLen, vLen = load_data(train_file, test_file)
    config['ulen'] = uLen
    config['vlen'] = vLen
    config['n_items'] = vLen
    p_dims = [200, 600, vLen]
    config['p_dims'] = p_dims
    print("用户项目数：", uLen, vLen)
    print("创建初始W,W2矩阵")
    W = create_RMN(train_data, uLen, vLen)
    W2 = create_RMN(test_data, uLen, vLen)

    m = None
    n = None

    replaced_indices = []

    u_vector, v_vector = init_vectors(rank, uLen, vLen)
    u_vectors = torch.from_numpy(u_vector).float().to(device)
    v_vectors = torch.from_numpy(v_vector).float().to(device)
    u_vectors = torch.nn.Parameter(u_vectors)
    v_vectors = torch.nn.Parameter(v_vectors)

    model = Net(config, u_vectors, v_vectors).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    R_fake_path = r'R_'+dataset+'.pt'
    z_path = r'R_vae_'+dataset+'.pt'

    R_fake = W
    z = W

    if os.path.exists(R_fake_path):
        checkpoint = torch.load(R_fake_path)
        R_fake = checkpoint['R_fake']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("载入训练矩阵R.")

    else:
        print("重新训练矩阵R.")
        R_fake = train_gnn(config, model, optimizer, R_fake, W2)
        torch.save({
            'R_fake': R_fake,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, R_fake_path)
    torch.save(model.u.weight, "user.pt")
    torch.save(model.v.weight, "item.pt")

    print("重新训练矩阵z.")
    R_fake = R_fake.detach().cpu()
    R_fake = R_fake.to(device)
    z = train_vae(config, model, optimizer, R_fake, W2)
