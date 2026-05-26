import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

# --- Physics Constants ---
JET_PT_MIN = 30.0
LEP_PT_MIN = 20.0

def calculate_mlj(pt1, eta1, phi1, pt2, eta2, phi2):
    """Calculates invariant mass assuming massless objects for simplicity"""
    return np.sqrt(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)))

def main():
    print("Generating Lepton-Jet Invariant Mass Plot...")
    
    # Point this to your ttbar ROOT file
    ttbar_file = "/DATA/akshat_b22258/MG5_aMC_v3_7_0/ttbarfinal/Events/run_01/tag_2_delphes_events.root"
    
    with uproot.open(ttbar_file) as f:
        tree = f["Delphes"]
        events = tree.arrays(["Electron.PT", "Electron.Eta", "Electron.Phi",
                              "Muon.PT", "Muon.Eta", "Muon.Phi",
                              "Jet.PT", "Jet.Eta", "Jet.Phi"])
        
    # Reconstruct objects
    e_pt, e_eta, e_phi = events["Electron.PT"], events["Electron.Eta"], events["Electron.Phi"]
    m_pt, m_eta, m_phi = events["Muon.PT"], events["Muon.Eta"], events["Muon.Phi"]
    j_pt, j_eta, j_phi = events["Jet.PT"], events["Jet.Eta"], events["Jet.Phi"]
    
    # Filter basic cuts
    good_e = (e_pt > LEP_PT_MIN)
    good_m = (m_pt > LEP_PT_MIN)
    good_j = (j_pt > JET_PT_MIN)
    
    # Flatten everything to event level for quick combination
    lep_pt = ak.concatenate([e_pt[good_e], m_pt[good_m]], axis=1)
    lep_eta = ak.concatenate([e_eta[good_e], m_eta[good_m]], axis=1)
    lep_phi = ak.concatenate([e_phi[good_e], m_phi[good_m]], axis=1)
    
    jet_pt = j_pt[good_j]
    jet_eta = j_eta[good_j]
    jet_phi = j_phi[good_j]
    
    # We only want events with at least 1 lepton and 1 jet
    mask = (ak.num(lep_pt) > 0) & (ak.num(jet_pt) > 0)
    lep_pt, lep_eta, lep_phi = lep_pt[mask], lep_eta[mask], lep_phi[mask]
    jet_pt, jet_eta, jet_phi = jet_pt[mask], jet_eta[mask], jet_phi[mask]
    
    # Calculate ALL lepton-jet combinations using Awkward Cartesian product
    leps = ak.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi})
    jets = ak.zip({"pt": jet_pt, "eta": jet_eta, "phi": jet_phi})
    
    pairs = ak.cartesian([leps, jets])
    l, j = ak.unzip(pairs)
    
    # Calculate mass
    mlj = calculate_mlj(l.pt, l.eta, l.phi, j.pt, j.eta, j.phi)
    mlj_flat = ak.flatten(mlj) # Flatten for plotting
    
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    
    # Plot histogram
    plt.hist(mlj_flat, bins=60, range=(0, 300), color='#4C72B0', edgecolor='black', alpha=0.8)
    
    # Add physics markers
    plt.axvline(153, color='red', linestyle='--', linewidth=2, label='Kinematic Endpoint (153 GeV)')
    plt.axvline(180, color='orange', linestyle=':', linewidth=2, label='Control Region Threshold (180 GeV)')
    
    # Formatting
    plt.title(r'Lepton-Jet Invariant Mass ($m_{\ell j}$) in $t\bar{t}$ Events', fontsize=14)
    plt.xlabel(r'$m_{\ell j}$ [GeV]', fontsize=12)
    plt.ylabel('Combinations / 5 GeV', fontsize=12)
    plt.xlim(0, 300)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('mlj_distribution.pdf')
    print("Saved plot to mlj_distribution.pdf!")

if __name__ == "__main__":
    main()