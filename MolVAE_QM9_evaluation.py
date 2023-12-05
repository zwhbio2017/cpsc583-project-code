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
dataset = QM9(root='../data/QM9/')

z_dim = 8
m_dim = 5
b_dim = 4
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

z = np.random.normal(0, 1, size=(batch_size, z_dim))
z = torch.from_numpy(z).to(device).float()
edges_logits, nodes_logits = decoder(z)
(edges_hard, nodes_hard) = postprocess((edges_logits, nodes_logits))
edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
mols = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
        for e_, n_ in zip(edges_hard, nodes_hard)]

mols_example = random.sample(mols, 12)
Chem.Draw.IPythonConsole.ShowMols(mols_example)
