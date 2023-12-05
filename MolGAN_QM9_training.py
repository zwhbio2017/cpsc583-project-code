import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.aggr import MulAggregation
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from collections import defaultdict
from utils import MolecularMetrics

# Suppress rdkit error message and set torch seed
RDLogger.DisableLog('rdApp.*')
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)

# Define models (Generator and Discriminator)
class Generator(nn.Module):
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(Generator, self).__init__()
        self.activation_f = nn.Tanh()
        dense_layers = []
        for c0, c1 in zip([z_dim] + conv_dims[:-1], conv_dims):
            dense_layers.append(nn.Linear(c0, c1))
            dense_layers.append(nn.Dropout(dropout_rate))
            dense_layers.append(nn.Tanh())
        self.multi_dense_layer = nn.Sequential(*dense_layers)

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class Discriminator(nn.Module):
    def __init__(self, conv_dim, m_dim, b_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(Discriminator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        self.dropout_rate = dropout_rate
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        
        self.conv1 = GCNConv(m_dim, graph_conv_dim[0])
        if self.dropout_rate > 0:
            self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = GCNConv(graph_conv_dim[0], graph_conv_dim[1])
        if self.dropout_rate > 0:
            self.dropout2 = nn.Dropout(dropout_rate)
        self.aggr = MulAggregation()
        
        dense_layers = []
        for c0, c1 in zip([graph_conv_dim[1]] + linear_dim[:-1], linear_dim):
            dense_layers.append(nn.Linear(c0, c1))
            dense_layers.append(nn.Dropout(dropout_rate))
            dense_layers.append(nn.Tanh())
        self.multi_dense_layer = nn.Sequential(*dense_layers)
        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, x, edge_index, activation=None):
        x = self.conv1(x, edge_index)
        x = self.activation_f(x)
        if self.dropout_rate > 0:
            x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.activation_f(x)
        if self.dropout_rate > 0:
            x = self.dropout2(x)
        x = self.aggr(x)
        x = self.multi_dense_layer(x)
        output = self.output_layer(x)
        if activation is not None:
            output = activation(output)

        return output, x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = QM9(root='../data/QM9/')

# Hyperparameter setting
z_dim = 8
m_dim = 5
b_dim = 4
g_conv_dim = [128, 256, 512]
d_conv_dim = [[128, 64], 128, [128, 64]]
la = 0
lambda_rec = 10
la_gp = 10
post_method = 'softmax'
metric = 'validity,qed'
test_size = 0.2
batch_size = 32
num_epochs = 100
num_steps = (len(dataset) // batch_size)
g_lr = 0.001
d_lr = 0.001
dropout = 0
n_critic = 1
resume_epoch = None

# Get all possible atoms and bond types from QM9 dataset
all_mols = [Chem.MolFromSmiles(smile) for smile in dataset.smiles]
atom_labels = set()
for idx, mol in enumerate(all_mols):
    if mol is not None:
        atoms = mol.GetAtoms()
        for atom in atoms:
            atom_labels.add(atom.GetAtomicNum())
atom_labels.add(0)
atom_labels = sorted(atom_labels)
atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
atom_num_types = len(atom_labels)

bond_labels = set()
for idx, mol in enumerate(all_mols):
    if mol is not None:
        bonds = mol.GetBonds()
        for bond in bonds:
            bond_labels.add(bond.GetBondType())
bond_labels.add(Chem.rdchem.BondType.ZERO)
bond_labels = sorted(bond_labels)
bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
bond_num_types = len(bond_labels)

# Set models and optimizers

G = Generator(g_conv_dim, z_dim, 10, b_dim, m_dim, dropout)
D = Discriminator(d_conv_dim, m_dim, b_dim - 1, dropout)
V = Discriminator(d_conv_dim, m_dim, b_dim - 1, dropout)

g_optimizer = torch.optim.RMSprop(G.parameters(), g_lr)
d_optimizer = torch.optim.RMSprop(D.parameters(), d_lr)
v_optimizer = torch.optim.RMSprop(V.parameters(), g_lr)

G.to(device)
D.to(device)
V.to(device)

train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=123)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Utility functions

def postprocess(inputs, temperature=1):
    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    softmax = [F.softmax(e_logits / temperature, -1) for e_logits in listify(inputs)]
    return [delistify(e) for e in (softmax)]

def gradient_penalty(y, x):
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)

def reward_mols(mols):
    rr = 1.
    for m in ('logp,sas,qed,unique' if metric == 'all' else metric).split(','):

        if m == 'np':
            rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
        elif m == 'logp':
            rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
        elif m == 'sas':
            rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
        elif m == 'qed':
            rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
        elif m == 'novelty':
            rr *= MolecularMetrics.novel_scores(mols, dataset)
        elif m == 'dc':
            rr *= MolecularMetrics.drugcandidate_scores(mols, dataset)
        elif m == 'unique':
            rr *= MolecularMetrics.unique_scores(mols)
        elif m == 'diversity':
            rr *= MolecularMetrics.diversity_scores(mols, dataset)
        elif m == 'validity':
            rr *= MolecularMetrics.valid_scores(mols)
        else:
            raise RuntimeError('{} is not defined as a metric'.format(m))

    return rr.reshape(-1, 1)

def get_reward(n_hat, e_hat):
    (edges_hard, nodes_hard) = postprocess((e_hat, n_hat))
    edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
    mols = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
            for e_, n_ in zip(edges_hard, nodes_hard)]
    reward = torch.from_numpy(reward_mols(mols)).to(device)
    return reward

def matrices2mol(node_labels, edge_labels, strict=False):
    mol = Chem.RWMol()
    for node_label in node_labels:
        mol.AddAtom(Chem.Atom(atom_decoder_m[node_label]))
    for start, end in zip(*np.nonzero(edge_labels)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[edge_labels[start, end]])
    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None

    return mol

def association2edge(edges):
    mean_value = torch.mean(edges)
    source, target = [], []
    for edge in edges:
        for i in range(edge.shape[0]):
            for j in range(edge.shape[1]):
                if i < j:
                    continue
                mean_association = torch.mean(edge[i, j])
                if mean_association > mean_value:
                    source.append(i)
                    target.append(j)

    return torch.tensor([source, target], dtype=torch.long)

# Training function

def train(train_loader, epoch_i):
    losses = defaultdict(list)
    scores = defaultdict(list)
    
    for data in train_loader:
        # 1. Create data needed.
        data = data.to(device)
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        z = torch.from_numpy(z).to(device).float()

        # 2. Train discriminator
        logits_real, features_real = D(data.x[:, 0:5], data.edge_index)
        edges_logits, nodes_logits = G(z)
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits))
        edges_index_hat = association2edge(edges_hat).to(device)
        logits_fake, features_fake = D(nodes_hat, edges_index_hat)

        d_loss_real = torch.mean(logits_real)
        d_loss_fake = torch.mean(logits_fake)
        loss_D = -d_loss_real + d_loss_fake

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        v_optimizer.zero_grad()
        loss_D.backward()
        d_optimizer.step()

        # 3. Train generator
        edges_logits, nodes_logits = G(z)
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits))
        edges_index_hat = association2edge(edges_hat).to(device)
        logits_fake, features_fake = D(nodes_hat, edges_index_hat)

        value_logit_real, _ = V(data.x[:, 0:5], data.edge_index, torch.sigmoid)
        value_logit_fake, _ = V(nodes_hat, edges_index_hat, torch.sigmoid)
        f_loss = (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2
        
        mols = [Chem.MolFromSmiles(data[i].smiles) for i in range(len(data))]
        reward_r = torch.from_numpy(reward_mols(mols)).to(device)
        reward_f = get_reward(nodes_hat, edges_hat)

        loss_G = -logits_fake
        loss_V = torch.abs(value_logit_real - torch.mean(reward_r)) + torch.abs(torch.mean(value_logit_fake) - torch.mean(reward_f))
        loss_RL = -value_logit_fake

        loss_G = torch.mean(loss_G)
        loss_V = torch.mean(loss_V)
        loss_RL = torch.mean(loss_RL)
        losses['l_G'].append(loss_G.item())
        losses['l_RL'].append(loss_RL.item())
        losses['l_V'].append(loss_V.item())

        alpha = torch.abs(loss_G.detach() / loss_RL.detach()).detach()
        train_step_G = la * loss_G + (1 - la) * alpha * loss_RL
        train_step_V = loss_V

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        v_optimizer.zero_grad()
        train_step_G.backward(retain_graph=True)
        train_step_V.backward(retain_graph=True)
        g_optimizer.step()
        v_optimizer.step()

        # 4. Print log
        log = "Iteration [{}/{}]:".format(epoch_i + 1, num_epochs)
        is_first = True
        for tag, value in losses.items():
            if is_first:
                log += "\n{}: {:.2f}".format(tag, np.mean(value))
                is_first = False
            else:
                log += ", {}: {:.2f}".format(tag, np.mean(value))
        is_first = True
        for tag, value in scores.items():
            if is_first:
                log += "\n{}: {:.2f}".format(tag, np.mean(value))
                is_first = False
            else:
                log += ", {}: {:.2f}".format(tag, np.mean(value))
        print(log)

for i in range(num_epochs):
    train(train_loader, i)

torch.save(G.state_dict(), '../model/MolGAN_QM9_G.pth')
torch.save(D.state_dict(), '../model/MolGAN_QM9_D.pth')
torch.save(V.state_dict(), '../model/MolGAN_QM9_V.pth')
