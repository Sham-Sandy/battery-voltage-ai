# ============================================================
# HTGNN Voltage Prediction for New Material
# EXACT MATCH TO TRAINING PIPELINE
# Input: CIF file + discharge formula
# ============================================================

import os
import torch
import numpy as np
import joblib

from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool

from pymatgen.core import Structure, Composition
from mendeleev import element

# ================= CONFIG =================

OUT_DIR = "Jesus_oxidation_enriched19"
MODEL_PATH = os.path.join(OUT_DIR, "htgnn_enriched19.pt")
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")
CAP_PATH = os.path.join(OUT_DIR, "capacity_cal_surr.pkl")

CUTOFF = 4.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSITION_METALS = set(range(21,31)) | set(range(39,49)) | set(range(72,81))
ANIONS = {8,9,16,17,34,35}
FARADAY = 96485

# ============================================================
# LOAD TRAINED OBJECTS
# ============================================================

scaler = joblib.load(SCALER_PATH)
cap_model = joblib.load(CAP_PATH)
TAB_DIM = scaler.mean_.shape[0]

# ============================================================
# MODEL (IDENTICAL TO TRAINING)
# ============================================================

class HTGNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.convs = nn.ModuleList([
            TransformerConv(3,64,edge_dim=1,heads=2),
            TransformerConv(128,64,edge_dim=1,heads=2),
            TransformerConv(128,64,edge_dim=1,heads=2),
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(3)])

        self.tab = nn.Sequential(
            nn.Linear(d,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU()
        )

        self.fuse = nn.Sequential(
            nn.Linear(256,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,g):
        h = g.x
        for c,n in zip(self.convs,self.norms):
            h = torch.relu(n(c(h,g.edge_index,g.edge_attr)))
        hg = global_mean_pool(h,g.batch)
        ht = self.tab(g.u)
        return self.fuse(torch.cat([hg,ht],1)).squeeze(-1)

# ============================================================
# ELEMENT CACHE (IDENTICAL LOGIC)
# ============================================================

ELEMENT_CACHE = {}

for Z in range(1,95):
    try:
        e = element(Z)
        ELEMENT_CACHE[e.symbol] = {
            "Z": e.atomic_number,
            "chi": e.en_pauling or 0,
            "radius": e.atomic_radius or 0,
            "IE": e.ionenergies.get(1,0) if e.ionenergies else 0,
            "valence": e.nvalence() or 0,
            "mass": e.atomic_weight
        }
    except:
        pass

# ============================================================
# THEORY CAPACITY
# ============================================================

def compute_theory_capacity(formula):
    M = Composition(formula).weight
    return FARADAY / (3.6 * M)

# ============================================================
# COMPOSITION DESCRIPTORS (EXACT ORDER)
# ============================================================

def composition_descriptors(formula):

    comp = Composition(formula)
    el_amt = comp.get_el_amt_dict()
    elements = list(el_amt.keys())
    amounts = np.array(list(el_amt.values()), dtype=float)
    fractions = amounts / amounts.sum()

    Z, chi, radius, IE, valence, mass = [], [], [], [], [], []

    for el in elements:
        props = ELEMENT_CACHE[el]
        Z.append(props["Z"])
        chi.append(props["chi"])
        radius.append(props["radius"])
        IE.append(props["IE"])
        valence.append(props["valence"])
        mass.append(props["mass"])

    Z = np.array(Z)
    chi = np.array(chi)
    radius = np.array(radius)
    IE = np.array(IE)
    valence = np.array(valence)
    mass = np.array(mass)

    def wmean(x): return np.sum(x * fractions)
    def wstd(x):  return np.sqrt(np.sum(fractions * (x - wmean(x))**2))

    chi_range = chi.max() - chi.min()
    radius_range = radius.max() - radius.min()

    tm_mask = np.isin(Z, list(TRANSITION_METALS))
    tm_fraction_comp = np.sum(fractions[tm_mask])
    tm_mean_chi = np.mean(chi[tm_mask]) if tm_mask.any() else 0

    return np.array([
        wmean(chi), wstd(chi),
        wmean(radius),
        wmean(IE),
        wmean(valence),
        wmean(mass),
        len(elements),
        tm_fraction_comp,
        chi_range,
        radius_range,
        wstd(valence),
        wstd(radius),
        tm_mean_chi
    ], dtype=np.float32)

# ============================================================
# GRAPH BUILDER (IDENTICAL TO TRAINING)
# ============================================================

def build_graph_from_cif(cif_path):

    structure = Structure.from_file(cif_path)

    try:
        structure.add_oxidation_state_by_guess()
        ox_states = [site.specie.oxi_state for site in structure]
        ox_ok = True
    except:
        ox_ok = False

    x = []
    for i, site in enumerate(structure):

        el = element(site.specie.symbol)
        Z = el.atomic_number
        chi = el.en_pauling or 0

        if ox_ok:
            ox = float(ox_states[i])
        else:
            ox = 0.0

        x.append([Z, chi, ox])

    x = torch.tensor(x, dtype=torch.float32)

    neighbors = structure.get_all_neighbors(CUTOFF)

    src, dst, attr = [], [], []
    for i, nbrs in enumerate(neighbors):
        for nbr in nbrs:
            src.append(i)
            dst.append(nbr.index)
            attr.append([nbr.nn_distance])

    edge_index = torch.tensor([src,dst], dtype=torch.long)
    edge_attr  = torch.tensor(attr, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_voltage(cif_path, formula):

    model = HTGNN(TAB_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    g = build_graph_from_cif(cif_path)
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)

    # ---- TABULAR FEATURES ----

    cap_theory = compute_theory_capacity(formula)
    cap_surr = cap_model.predict([[cap_theory]])[0]

    comp_desc = composition_descriptors(formula)

    Z_nodes = g.x[:,0].numpy()
    tm = np.isin(Z_nodes, list(TRANSITION_METALS))
    an = np.isin(Z_nodes, list(ANIONS))

    redox = np.array([
        tm.mean(),
        Z_nodes[tm].mean() if tm.any() else 0.0,
        an.mean(),
        Z_nodes.mean(),
        Z_nodes.std()
    ], dtype=np.float32)

    x_tab = np.concatenate([[cap_surr], redox, comp_desc])
    x_tab_scaled = scaler.transform(x_tab.reshape(1,-1))

    g.u = torch.tensor(x_tab_scaled, dtype=torch.float32)
    g = g.to(DEVICE)

    with torch.no_grad():
        pred = model(g).cpu().item()

    return pred

# ============================================================
# USER INPUT
# ============================================================

if __name__ == "__main__":

    cif_file = input("Enter path to CIF file: ")
    formula = input("Enter discharge formula: ")

    voltage = predict_voltage(cif_file, formula)

    print("\n======================================")
    print("Predicted Average Voltage: {:.3f} V".format(voltage))
    print("======================================")
