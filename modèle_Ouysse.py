#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
from numba import njit
import xarray as xr
import os
import pandas as pd
import json
from datetime import datetime
import pe_oudin
from hydrogr import InputDataHandler, ModelGr4h
from meteofetch import Arpege01
from datetime import datetime

# =============================================================
# Fonctions génériques
# =============================================================
def save_states(filepath, states):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    json_states = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in states.items()}
    with open(filepath, 'w') as f:
        json.dump(json_states, f)

def load_states(filepath, default_states=None):
    if default_states is None:
        default_states = {}
    try:
        with open(filepath, 'r') as f:
            states = json.load(f)
        return {k: np.array(v) if isinstance(v, list) else v for k, v in states.items()}
    except:
        return default_states

def filter_by_datetime(df, LastQ_datetime):
    if LastQ_datetime:
        LastQ_datetime = pd.to_datetime(LastQ_datetime)
        return df[df.index >= LastQ_datetime]
    return df

def assimilate_flow(outputs, flow_col, last_q):
    if last_q and len(outputs) > 0 and outputs[flow_col].iloc[0] != 0:
        correction_factor = last_q / outputs[flow_col].iloc[0]
        outputs[flow_col] = outputs[flow_col] * correction_factor
    return outputs

# =============================================================
# Fonction pour récuperer les données Arpeges sur plusieurs polygones
# =============================================================
BASINS = {
    "Themines": {
        "indices": [(247, 338), (247, 339), (248, 338), (248, 339)],
        "weights": [0.043, 0.169, 0.212, 0.574]
    },
    "Alzou": {
        "indices": [(248, 337), (248, 338)],
        "weights": [0.275, 0.724]
    },
    "Theminettes": {
        "indices": [(247, 339), (247, 340), (248, 339), (248, 340)],
        "weights": [0.790, 0.104, 0.097, 0.007]
    },
    "Karst": {
        "indices": [(246, 336), (246, 337), (247, 335), (247, 336), (247, 337), (247, 338), (247, 339), (248, 335), (248, 336), (248, 337), (248, 338)],
        "weights": [0.020, 0.063, 0.034, 0.171, 0.193, 0.126, 0.004, 0.102, 0.159, 0.083, 0.039]
    }
}
def get_Arpege_data(indices, weights):
    INDICES = np.array(indices)
    WEIGHTS_ARRAY = np.array(weights)
    i_idx = INDICES[:, 0]
    j_idx = INDICES[:, 1]
    ds = Arpege01.get_latest_forecast(paquet='SP1', variables=('t2m','tp'))
    for k in ds:
        ds[k] = ds[k].drop_vars('step', errors='ignore')
    ds = xr.Dataset(ds)
    tp = np.dot(ds.tp.values[:, i_idx, j_idx], WEIGHTS_ARRAY)
    t2m = np.dot(ds.t2m.values[:, i_idx, j_idx], WEIGHTS_ARRAY) - 273.15
    
    # PE_Oudin
    times_list = pd.to_datetime(ds.time.values).to_pydatetime().tolist()
    ET = pe_oudin.PE_Oudin.pe_oudin(temp=t2m, time=times_list, 
                                   lat=44.74, lat_unit='deg', out_units='mm/hour')
    
    df = pd.DataFrame({
        'Date': ds.time.values,
        'precipitation': tp,
        'temperature': t2m,
        'evapotranspiration': np.array(ET)
    })
    df = df.dropna()
    return df
# =============================================================
# Modélisation débits des pertes (GR4H)
# =============================================================
GR4H_parameter = {
    "Themines": {"params": {"X1": 243.463, "X2": -0.193, "X3": 65.874, "X4": 3.102}, 
                 "surface": 55.62},
    "Alzou": {"params": {"X1": 200.0, "X2": -0.1, "X3": 70.0, "X4": 2.8}, 
              "surface": 85.3},
    "Theminettes": {"params": {"X1": 220.0, "X2": -0.15, "X3": 60.0, "X4": 3.5}, 
                    "surface": 42.1}
}
def run_gr4h(arpege_data, bassin_name, lastQ=None, LastQ_datetime=None):
    df = arpege_data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    model = ModelGr4h(GR4H_parameter[bassin_name]["params"])
    
    states_file = f"Etats_niveaux_réservoirs//{bassin_name.lower()}_states.json"
    states = load_states(states_file)
    if states: 
        model.set_states(states)
    
    GR4H_outputs = model.run(InputDataHandler(ModelGr4h, df).data)
    save_states(states_file, model.get_states())
    
    GR4H_outputs = filter_by_datetime(GR4H_outputs, LastQ_datetime)
    
    if lastQ: 
        lastQ_mm_h = lastQ * 3.6 / GR4H_parameter[bassin_name]["surface"]
        GR4H_outputs = assimilate_flow(GR4H_outputs, 'flow', lastQ_mm_h)
    
    GR4H_outputs['flow_m3_s'] = GR4H_outputs['flow'] / 3.6 * GR4H_parameter[bassin_name]["surface"]
    
    return GR4H_outputs

# =============================================================
# Modélisation débit éxutoire (KarstMod)
# =============================================================

# Conversion des unités
@njit()
def to_q_m3_s(q_mm_h=np.array([],dtype=np.float64), area_km2=np.float64(0)):
    """Conversion mm/h -> m³/s"""
    return q_mm_h / 24 * area_km2

@njit()
def to_q_mm_h(q_m3_s=np.array([],dtype=np.float64), area_km2=np.float64(0)):
    """Conversion m³/s -> mm/h"""
    return q_m3_s * 3.6 * 24 / area_km2

# Fonctions de base
@njit()
def ki_seuil(k=np.float64(0), a=np.float64(0), H=np.float64(0), Hseuil=np.float64(0)):
    return np.maximum(k * (H - Hseuil)**(a-1), 0)

@njit()
def Eth(E=np.float64(0), k=np.float64(0), S=np.float64(0), Emin=np.float64(0)):
    if k != 0:
        Eq = S / k
        return np.maximum(Eq + (E - Eq) * np.exp(-k), Emin)
    else:
        return np.maximum(E + S, Emin)

# Niveau superieur (Epikarst)
@njit()
def tf_E(pr=np.array([],dtype=np.float64), pet=np.array([],dtype=np.float64), Emin=np.float64(0),
         kEM=np.float64(0), aEM=np.float64(0), kEC=np.float64(0), aEC=np.float64(0),
         kES=np.float64(0), aES=np.float64(0), kloss=np.float64(0), aloss=np.float64(0), Eloss=np.float64(0),
         wl_initial=np.float64(0)):
    
    QEM   = np.zeros(len(pr)+1, np.float64)
    QEC   = np.zeros(len(pr)+1, np.float64)
    QES   = np.zeros(len(pr)+1, np.float64)
    Qloss = np.zeros(len(pr)+1, np.float64)
    wl    = np.zeros(len(pr)+1, np.float64)
    
    wl[0] = wl_initial
    
    for i in range(len(pr)):
        kEMi = ki_seuil(kEM, aEM, wl[i], Emin)
        kECi = ki_seuil(kEC, aEC, wl[i], Emin)
        kESi = ki_seuil(kES, aES, wl[i], Emin)
        klossi = ki_seuil(kloss, aloss, wl[i], Eloss)
        kE = kEMi + kECi + kESi + klossi
        
        SE = pr[i] - pet[i] - klossi*Eloss
        E12 = Eth(wl[i], kE, SE, Emin)
        
        kEMi = ki_seuil(kEM, aEM, E12, Emin)
        kECi = ki_seuil(kEC, aEC, E12, Emin)
        kESi = ki_seuil(kES, aES, E12, Emin)
        klossi = ki_seuil(kloss, aloss, E12, Eloss)
        kE = kEMi + kECi + kESi + klossi
        
        SE = pr[i] - pet[i] - klossi*Eloss
        wl[i+1] = Eth(wl[i], kE, SE, Emin)
        
        if kE != 0:
            Qtot = max(SE + (wl[i]-wl[i+1]), 0)
            Qloss[i] = max(klossi*(Qtot/kE - Eloss), 0)
            QES[i] = max(kESi*Qtot/kE, 0)
            QEC[i] = max(kECi*Qtot/kE, 0)
            QEM[i] = max(kEMi*Qtot/kE, 0)
            
    return QEM[:-1], QEC[:-1], QES[:-1], Qloss[:-1], wl[:-1], wl[-1]
    
# Niveau inférieur (Matrice & Conduit)
@njit()
def tf_MC(input_M=np.array([],dtype=np.float64), output_M=np.array([],dtype=np.float64),
          input_C=np.array([],dtype=np.float64), output_C=np.array([],dtype=np.float64),
          kMC=np.float64(0), aMC=np.float64(0), C_loss=np.float64(0), M_loss=np.float64(0),
          kMS=np.float64(0), aMS=np.float64(0), kCS=np.float64(0), aCS=np.float64(0),
          C_initial=np.float64(0), M_initial=np.float64(0)):
    
    C = np.zeros(len(input_M)+1, np.float64)
    M = np.zeros(len(input_M)+1, np.float64)
    Q_C_loss = np.zeros(len(input_M)+1, np.float64)
    Q_M_loss = np.zeros(len(input_M)+1, np.float64)
    Q_M_S    = np.zeros(len(input_M)+1, np.float64)
    Q_C_S    = np.zeros(len(input_M)+1, np.float64)
    
    C[0] = C_initial
    M[0] = M_initial
    
    SM = input_M - output_M
    SC = input_C - output_C
    
    for i in range(len(input_M)):
        M[i+1] = min(M[i]+SM[i], M_loss)
        C[i+1] = min(C[i]+SC[i], C_loss)
        Q_M_S[i] = SM[i]
        Q_C_S[i] = SC[i]
        Q_M_loss[i] = max(M[i]-M_loss, 0)
        Q_C_loss[i] = max(C[i]-C_loss, 0)
    
    qsim = np.maximum(Q_M_S + Q_C_S + Q_M_loss + Q_C_loss, 0)
    return qsim[:-1], C[:-1], M[:-1], C[-1], M[-1]

@njit()
def karstmod_engine(pr=np.array([],dtype=np.float64),
                pet=np.array([],dtype=np.float64),
                qsink_mm=np.array([],dtype=np.float64),
                area=np.float64(0),
                Emin=np.float64(0),
                kEM=np.float64(0), aEM=np.float64(0),
                kEC=np.float64(0), aEC=np.float64(0),
                kES=np.float64(0), aES=np.float64(0),
                kloss=np.float64(0), aloss=np.float64(0), Eloss=np.float64(0),
                kCS=np.float64(0), aCS=np.float64(0),
                kMS=np.float64(0), aMS=np.float64(0),
                kMC=np.float64(0), aMC=np.float64(0),
                wlE_initial=np.float64(0),
                C_initial=np.float64(0), 
                M_initial=np.float64(0)):

    qEM, qEC, qES, qloss, wlE, wlE_final = tf_E(pr, pet, Emin, kEM, aEM, kEC, aEC, kES, aES, kloss, aloss, Eloss, wlE_initial)
    
    input_M   = qEM
    output_M  = np.zeros(len(pr), dtype=np.float64)
    input_C   = qEC + qsink_mm
    output_C  = np.zeros(len(pr), dtype=np.float64)
    
    qCS, wl_C, wl_M, C_final, M_final = tf_MC(input_M=input_M, output_M=output_M, input_C=input_C, output_C=output_C,
                      kMC=kMC, aMC=aMC, C_loss=1e5, M_loss=1e5, kMS=kMS, aMS=aMS, kCS=kCS, aCS=aCS,
                      C_initial=C_initial, M_initial=M_initial)
    
    qsim = to_q_m3_s(qCS, area)
    
    return qsim, wlE_final, C_final, M_final

def run_karstmod(arpege_data, qsink_data, 
             LastQ_datetime=None,
             params_file="Parametres_Karstmod/params_ouysse.csv",
             lastQ=None):
    
    states_file = os.path.join("Etats_niveaux_réservoirs/karstmod_states.json")
    
    df = arpege_data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    df = filter_by_datetime(df, LastQ_datetime)
    
    params = pd.read_csv(params_file, sep=";", header=None, index_col=0)
    RA = float(params.loc["RA (km2)"].values[0])
    kCS = float(params.loc["kCS (mm/hour)"].values[0])
    kMS = float(params.loc["kMS (mm/hour)"].values[0])
    kMC = float(params.loc["kMC (mm/hour)"].values[0])
    kEM = float(params.loc["kEM (mm/hour)"].values[0])
    kEC = float(params.loc["kEC (mm/hour)"].values[0])
    alphaMS = float(params.loc["alphaMS"].values[0])
    alphaMC = float(params.loc["alphaMC"].values[0])
    
    Qsink_mm_h = to_q_mm_h(np.array(qsink_data, dtype=np.float64), RA)
    
    states = load_states(states_file, default_states={'wlE_final':0.0, 'C_final':0.0, 'M_final':0.0})
    
    qsim, wlE_final, C_final, M_final = karstmod_engine(pr=np.array(df['precipitation'], dtype=np.float64),
                       pet=np.array(df['evapotranspiration'], dtype=np.float64),
                       qsink_mm=Qsink_mm_h,
                       area=RA,
                       Emin=-15, kEM=kEM, aEM=1, kEC=kEC, aEC=1,
                       kES=0, aES=1, kloss=0, aloss=1, Eloss=1e5,
                       kCS=kCS, aCS=1, kMS=kMS, aMS=alphaMS,
                       kMC=kMC, aMC=alphaMC,
                       wlE_initial=states['wlE_final'],
                       C_initial=states['C_final'],
                       M_initial=states['M_final'])
    
    new_states = {
        'wlE_final': float(wlE_final),
        'C_final': float(C_final),
        'M_final': float(M_final)
    }
    save_states(states_file, new_states)
    
    Karstmod_outputs = pd.DataFrame(index=df.index)
    Karstmod_outputs['flow_m3_s'] = qsim
    
    if lastQ:
        Karstmod_outputs = assimilate_flow(Karstmod_outputs, 'flow_m3_s', lastQ)
    
    return Karstmod_outputs


# In[22]:


# ============================================================
# Lancer la prévision 
# ============================================================
if __name__ == "__main__":

    datetime_now =datetime.now().replace(microsecond=0, second=0, minute=0) #doit être remplacer par synapse
    
# Récupération prévision Arpege
    Arpege_data = {name: get_Arpege_data(cfg["indices"], cfg["weights"]) 
                    for name, cfg in BASINS.items()}

# Exécution GR4H
    gr4h_Themines = run_gr4h(Arpege_data["Themines"], "Themines", 
                            lastQ=2.28, LastQ_datetime=datetime_now)
    gr4h_Alzou = run_gr4h(Arpege_data["Alzou"], "Alzou", 
                        lastQ=2.28, LastQ_datetime=datetime_now)
    gr4h_Theminettes = run_gr4h(Arpege_data["Theminettes"], "Theminettes", 
                            LastQ_datetime=datetime_now)
    
    print("GR4H Themines:")
    print(gr4h_Themines.head())
    
# Calcul du débit des pertes cumulé
    Qsink_sum = (
        (gr4h_Themines['flow_m3_s'].values +
        gr4h_Alzou['flow_m3_s'].values)
        *1.2 # +gr4h_Theminettes['flow_m3_s'].values (pas encore disponible sur la plateforme 
    )
    
# Exécution Karstmod
    Karstmod_ouysse = run_karstmod(
        arpege_data=Arpege_data["Karst"],
        qsink_data=Qsink_sum,
        LastQ_datetime=datetime_now, 
        lastQ=2.57
    )
    
    print("Karstmod Ouysse:")
    print(Karstmod_ouysse.head())


# In[ ]:




