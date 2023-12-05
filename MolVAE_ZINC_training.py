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

# Define models (Decoder, Encoder, and Discriminator)
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

class EncoderVAE(nn.Module):
    def __init__(self, conv_dim, m_dim, b_dim, z_dim, with_features=False, f_dim=0, dropout_rate=0):
        super(EncoderVAE, self).__init__()
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
        self.batch_layer = nn.Linear(linear_dim[-1], linear_dim[-1] * batch_size)
        self.emb_mean = nn.Linear(linear_dim[-1], z_dim)
        self.emb_logvar = nn.Linear(linear_dim[-1], z_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

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
        x = self.batch_layer(x).view(batch_size, -1)
        x_mu = self.emb_mean(x)
        x_logvar = self.emb_logvar(x)
        x = self.reparameterize(x_mu, x_logvar)
        return x, x_mu, x_logvar

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
dataset = ZINC(root='../data/ZINC/')

# Hyperparameter setting
z_dim = 8
m_dim = 28
b_dim = 5
g_conv_dim = [128, 256, 512]
d_conv_dim = [[128, 64], 128, [128, 64]]
la = 0
lambda_rec = 10
la_gp = 10
la_kl = 1
post_method = 'softmax'
metric = 'validity,qed'
test_size = 0.2
batch_size = 32
num_epochs = 50
num_steps = (len(dataset) // batch_size)
g_lr = 0.001
d_lr = 0.001
dropout = 0
n_critic = 1
resume_epoch = None

# Convert ZINC data to SMILES for molecule generation
def zinc2smiles(data):
    atom_map = [6, 8, 7, 9, 6, 16, 17, 8, 7, 35, 7, 7, 7, 7, 16, 53, 15, 8, 7, 7, 16, 15, 15, 6, 15, 16, 6, 15]
    charge_map = [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 1, 1, -1, -1, 0, 0, 1, -1, 1, 1, 0, 0, -1, 1, 1, -1, 1]
    xmap = {'C': 0, 
        'O': 1, 
        'N': 2, 
        'F': 3, 
        'C H1': 4, 
        'S': 5, 
        'Cl': 6, 
        'O -': 7, 
        'N H1 +': 8, 
        'Br': 9, 
        'N H3 +': 10, 
        'N H2 +': 11, 
        'N +': 12, 
        'N -': 13, 
        'S -': 14, 
        'I': 15, 
        'P': 16, 
        'O H1 +': 17, 
        'N H1 -': 18, 
        'O +': 19, 
        'S +': 20, 
        'P H1': 21, 
        'P H2': 22, 
        'C H2 -': 23, 
        'P +': 24, 
        'S H1 +': 25, 
        'C H1 -': 26, 
        'P H1 +': 27}
    
    mol = Chem.RWMol()

    for i in range(data.num_nodes):
        atom = Chem.Atom(atom_map[data.x[i, 0].item()])
        atom.SetFormalCharge(charge_map[data.x[i, 0].item()])
        mol.AddAtom(atom)

    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue

        bond_type = Chem.BondType.values[data.edge_attr[i].item()]
        mol.AddBond(src, dst, bond_type)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return Chem.MolToSmiles(mol, isomericSmiles=True)

# Set models and optimizers
decoder = Generator(g_conv_dim, z_dim, 38, b_dim, m_dim, dropout)
encoder = EncoderVAE(d_conv_dim, m_dim, b_dim - 1, z_dim, dropout_rate=dropout)
V = Discriminator(d_conv_dim, m_dim, b_dim - 1, dropout_rate=dropout)

vae_optimizer = torch.optim.RMSprop(list(decoder.parameters()) + list(encoder.parameters()) , g_lr)
v_optimizer = torch.optim.RMSprop(V.parameters(), d_lr)

decoder.to(device)
encoder.to(device)
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
    atom_map = [6, 8, 7, 9, 6, 16, 17, 8, 7, 35, 7, 7, 7, 7, 16, 53, 15, 8, 7, 7, 16, 15, 15, 6, 15, 16, 6, 15]
    charge_map = [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 1, 1, -1, -1, 0, 0, 1, -1, 1, 1, 0, 0, -1, 1, 1, -1, 1]
    xmap = {'C': 0, 
        'O': 1, 
        'N': 2, 
        'F': 3, 
        'C H1': 4, 
        'S': 5, 
        'Cl': 6, 
        'O -': 7, 
        'N H1 +': 8, 
        'Br': 9, 
        'N H3 +': 10, 
        'N H2 +': 11, 
        'N +': 12, 
        'N -': 13, 
        'S -': 14, 
        'I': 15, 
        'P': 16, 
        'O H1 +': 17, 
        'N H1 -': 18, 
        'O +': 19, 
        'S +': 20, 
        'P H1': 21, 
        'P H2': 22, 
        'C H2 -': 23, 
        'P +': 24, 
        'S H1 +': 25, 
        'C H1 -': 26, 
        'P H1 +': 27}
    
    mol = Chem.RWMol()

    for i in range(len(node_labels)):
        atom = Chem.Atom(atom_map[node_labels[i].item()])
        atom.SetFormalCharge(charge_map[node_labels[i].item()])
        mol.AddAtom(atom)

    visited = set()

    for i in range(edge_labels.shape[0]):
        for j in range(i + 1, edge_labels.shape[1]):
            bond_type = Chem.BondType.values[edge_labels[i, j].item()]
            mol.AddBond(i, j, bond_type)

    mol = mol.GetMol()

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

def x2nodes(data):
    x = label2onehot(data.x, m_dim).squeeze(1)
    list_x = []
    for i in range(len(data)):
        list_x.append(torch.mean(x[data.batch == i, ], dim=0))
    return torch.stack(list_x, dim=0)

def edges2association(data):
    edges_real = torch.zeros(32, 38, 38, 5).to(device)
    edge_attr = label2onehot(data.edge_attr, b_dim).squeeze(1)
    for i in range(data.edge_index.shape[1]):
        node1, node2 = data.edge_index[:, i]
        idx1 = torch.where(torch.where(data.batch == data.batch[node1])[0] == node1)[0]
        idx2 = torch.where(torch.where(data.batch == data.batch[node2])[0] == node2)[0]
        edges_real[data.batch[node1], min(idx1, 27), min(idx2, 27), ] = edge_attr[i]

    return edges_real

def label2onehot(labels, dim):
    out = torch.zeros(list(labels.size()) + [dim]).to(device)
    out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
    return out

# Training function

def train(train_loader, epoch_i):
    losses = defaultdict(list)
    
    for data in train_loader:
        # 1. Create data needed.
        data = data.to(device)
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        z = torch.from_numpy(z).to(device).float()

        # 2. Train VAE model
        x, x_mu, x_logvar = encoder(label2onehot(data.x, m_dim).squeeze(1), data.edge_index)
        edges_logits, nodes_logits = decoder(x)
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits))

        nodes_real = x2nodes(data)
        n_loss = torch.nn.CrossEntropyLoss(reduction='mean')(torch.mean(nodes_hat, dim=1).view(-1, m_dim)[:nodes_real.shape[0], ], nodes_real)
        edges_real = edges2association(data)
        e_loss = torch.nn.CrossEntropyLoss(reduction='mean')(edges_hat.reshape((-1, b_dim)), edges_real.reshape((-1, b_dim)))
        kl_loss = torch.mean(-0.5 * torch.sum(1 + x_logvar - x_mu ** 2 - x_logvar.exp(), dim=1), dim=0)
        loss_vae = n_loss + e_loss + la_kl * kl_loss
        
        mols = [Chem.MolFromSmiles(zinc2smiles(data[i])) for i in range(len(data))]
        reward_r = torch.from_numpy(reward_mols(mols)).to(device)
        reward_f = get_reward(nodes_logits, edges_logits)

        edges_index_hat = association2edge(edges_hat).to(device)
        value_logit_real, _ = V(label2onehot(data.x, m_dim).squeeze(1), data.edge_index, torch.sigmoid)
        value_logit_fake, _ = V(nodes_hat, edges_index_hat, torch.sigmoid)

        loss_V = torch.mean((value_logit_real - torch.mean(reward_r)) ** 2 * len(data) + (value_logit_fake.squeeze(1) - reward_f) ** 2)
        loss_rl = torch.mean(-value_logit_fake)
        alpha = torch.abs(loss_vae.detach() / loss_rl.detach())
        loss_rl = alpha * loss_rl
        vae_loss_train = la * loss_vae + (1 - la) * loss_rl

        losses['l_Rec'].append((n_loss + e_loss).item())
        losses['l_KL'].append(kl_loss.item())
        losses['l_VAE'].append(loss_vae.item())
        losses['l_RL'].append(loss_rl.item())
        losses['l_V'].append(loss_V.item())

        vae_optimizer.zero_grad()
        v_optimizer.zero_grad()
        vae_loss_train.backward(retain_graph=True)
        loss_V.backward(retain_graph=True)
        vae_optimizer.step()
        v_optimizer.step()

        # 3. Print log
        log = "Iteration [{}/{}]:".format(epoch_i + 1, num_epochs)
        is_first = True
        for tag, value in losses.items():
            if is_first:
                log += "\n{}: {:.2f}".format(tag, np.mean(value))
                is_first = False
            else:
                log += ", {}: {:.2f}".format(tag, np.mean(value))
        print(log)

for i in range(num_epochs):
    train(train_loader, i)

torch.save(decoder.state_dict(), '../model/MolVAE_ZINC_decoder.pth')
torch.save(encoder.state_dict(), '../model/MolVAE_ZINC_encoder.pth')
torch.save(V.state_dict(), '../model/MolVAE_ZINC_V.pth')
