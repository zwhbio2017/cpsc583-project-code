import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.datasets import QM9, ZINC
import random

RDLogger.DisableLog('rdApp.*')

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = ZINC(root='../data/ZINC/')

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
num_epochs = 100
num_steps = (len(dataset) // batch_size)
g_lr = 0.001
d_lr = 0.001
dropout = 0
n_critic = 1
resume_epoch = None

decoder = Generator(g_conv_dim, z_dim, 10, b_dim, m_dim, dropout)
state_dict = torch.load('../model/MolVAE_QM9_decoder.pth')
decoder.load_state_dict(state_dict)
decoder.to(device)

def postprocess(inputs, temperature=1):
    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    softmax = [F.softmax(e_logits / temperature, -1) for e_logits in listify(inputs)]
    return [delistify(e) for e in (softmax)]

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

    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None

    return mol

z = np.random.normal(0, 1, size=(1000, z_dim))
z = torch.from_numpy(z).to(device).float()
edges_logits, nodes_logits = decoder(z)
(edges_hard, nodes_hard) = postprocess((edges_logits, nodes_logits))
edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
mols = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
        for e_, n_ in zip(edges_hard, nodes_hard)]

mols_example = random.sample(mols, 12)
Chem.Draw.IPythonConsole.ShowMols(mols_example)
