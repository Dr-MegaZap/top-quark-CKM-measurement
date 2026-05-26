import argparse
import pickle
import numpy as np
import uproot
import awkward as ak

# --- Constants from the Paper ---
JET_PT_MIN   = 30.0    
JET_ETA_MAX  = 2.4
LEP_PT_MIN   = 20.0
LEP_ETA_MAX  = 2.4
DR_LEP_JET   = 0.4     

# Tagger Proxies (Table I)
# b-tag: Uses Delphes built-in BTag (which typically requires tracks/impact parameters)
# q-tag: No b-tag AND low track multiplicity (NCharged < 20)
N_CHARGED_Q_MAX = 20

def get_counts(events):
    # 0. Reconstruct physics objects
    electrons = ak.zip({"PT": events["Electron.PT"], "Eta": events["Electron.Eta"], "Phi": events["Electron.Phi"], "Charge": events["Electron.Charge"]})
    muons = ak.zip({"PT": events["Muon.PT"], "Eta": events["Muon.Eta"], "Phi": events["Muon.Phi"], "Charge": events["Muon.Charge"]})
    
    # We include BTag and NCharged for the Table II fitting logic
    jets = ak.zip({
        "PT": events["Jet.PT"], 
        "Eta": events["Jet.Eta"], 
        "Phi": events["Jet.Phi"],
        "BTag": events["Jet.BTag"],
        "NCharged": events["Jet.NCharged"]
    })
    
    met = ak.sum(events["MissingET.MET"], axis=1)

    # 1. Lepton Selection
    good_e = electrons[(electrons.PT > LEP_PT_MIN) & (abs(electrons.Eta) < LEP_ETA_MAX)]
    good_m = muons[(muons.PT > LEP_PT_MIN) & (abs(muons.Eta) < LEP_ETA_MAX)]
    
    dilep_mask = (ak.num(good_e) + ak.num(good_m) == 2)

    ge = good_e[dilep_mask]
    gm = good_m[dilep_mask]
    jets_filt = jets[dilep_mask]
    met_filt = met[dilep_mask]
    
    # 2. Channel Identification
    ee_mask   = (ak.num(ge) == 2) & (ak.sum(ge.Charge, axis=1) == 0)
    mumu_mask = (ak.num(gm) == 2) & (ak.sum(gm.Charge, axis=1) == 0)
    emu_mask  = (ak.num(ge) == 1) & (ak.num(gm) == 1) & (ak.sum(ge.Charge, axis=1) + ak.sum(gm.Charge, axis=1) == 0)

    # Data structures to match Step 2 and Step 4
    results_nj = np.zeros((3, 3)) # Figure 2
    results_nb_nq = {0: {}, 1: {}, 2: {}} # Table II matrices
    for ll in range(3):
        for nj_val in [2, 3, 4]:
            results_nb_nq[ll][nj_val] = np.zeros((nj_val + 1, nj_val + 1))

    # 3. Physics Cuts & Tagging
    for ch_idx, ch_mask in enumerate([ee_mask, mumu_mask, emu_mask]):
        if not ak.any(ch_mask): continue
            
        c_ge, c_gm, c_jets, c_met = ge[ch_mask], gm[ch_mask], jets_filt[ch_mask], met_filt[ch_mask]
        
        # Physics Vetoes (Z-peak and MET)
        if ch_idx in [0, 1]: 
            leps = c_ge if ch_idx == 0 else c_gm
            mll = np.sqrt(2 * leps.PT[:, 0] * leps.PT[:, 1] * (np.cosh(leps.Eta[:, 0] - leps.Eta[:, 1]) - np.cos(leps.Phi[:, 0] - leps.Phi[:, 1])))
            veto_mask = (np.abs(mll - 91.2) > 15.0) & (c_met > 40.0)
            c_ge, c_gm, c_jets = c_ge[veto_mask], c_gm[veto_mask], c_jets[veto_mask]
            
        if len(c_jets) == 0: continue
        
        # Overlap Removal
        leps_eta = ak.concatenate([c_ge.Eta, c_gm.Eta], axis=1)
        leps_phi = ak.concatenate([c_ge.Phi, c_gm.Phi], axis=1)
        
        def delta_r(eta1, phi1, eta2, phi2):
            return np.sqrt((eta1 - eta2)**2 + ((phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi)**2)

        dr1 = delta_r(c_jets.Eta, c_jets.Phi, leps_eta[:, 0], leps_phi[:, 0])
        dr2 = delta_r(c_jets.Eta, c_jets.Phi, leps_eta[:, 1], leps_phi[:, 1])
        
        # Analysis Jet Mask
        clean_jets_mask = (c_jets.PT > JET_PT_MIN) & (abs(c_jets.Eta) < JET_ETA_MAX) & (dr1 > DR_LEP_JET) & (dr2 > DR_LEP_JET)
        
        # Apply the taggers (Orthogonal)
        is_btag = (c_jets.BTag == 1)
        is_qtag = (~is_btag) & (c_jets.NCharged < N_CHARGED_Q_MAX)
        
        # Get final jet counts and tag counts per event
        final_nj = ak.num(c_jets[clean_jets_mask])
        final_nb = ak.sum(is_btag[clean_jets_mask], axis=1)
        final_nq = ak.sum(is_qtag[clean_jets_mask], axis=1)
        
        for n_j in [2, 3, 4]:
            results_nj[ch_idx, n_j-2] = ak.sum(final_nj == n_j)
            
            # Populate nb_nq matrices for events with exactly n_j jets
            ev_n_j = (final_nj == n_j)
            nb_in_ev = final_nb[ev_n_j]
            nq_in_ev = final_nq[ev_n_j]
            
            for b, q in zip(nb_in_ev, nq_in_ev):
                if b <= n_j and q <= (n_j - b):
                    results_nb_nq[ch_idx][n_j][b, q] += 1

    return results_nj, results_nb_nq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--nevt", type=int, required=True)
    parser.add_argument("--lumi", type=float, default=19.7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    final_nj = np.zeros((3, 3))
    final_nb_nq = {0: {2:np.zeros((3,3)), 3:np.zeros((4,4)), 4:np.zeros((5,5))},
                   1: {2:np.zeros((3,3)), 3:np.zeros((4,4)), 4:np.zeros((5,5))},
                   2: {2:np.zeros((3,3)), 3:np.zeros((4,4)), 4:np.zeros((5,5))}}

    for fp in args.files:
        with uproot.open(fp) as f:
            tree = f["Delphes"]
            # branches we need for physics + tagging
            branches = ["Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge",
                        "Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.Charge",
                        "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.BTag", "Jet.NCharged",
                        "MissingET.MET"]
            events = tree.arrays(branches)
            nj_arr, nbnq_dict = get_counts(events)
            final_nj += nj_arr
            for ll in range(3):
                for nj_v in [2,3,4]:
                    final_nb_nq[ll][nj_v] += nbnq_dict[ll][nj_v]

    scale = (args.sigma * args.lumi * 1000.0) / args.nevt
    
    # Scale everything
    for ll in range(3):
        final_nj[ll] *= scale
        for nj_v in [2,3,4]:
            final_nb_nq[ll][nj_v] *= scale

    out_dict = {
        "counts": {"nj": {0: final_nj[0], 1: final_nj[1], 2: final_nj[2]}, "nb_nq": final_nb_nq},
        "lumi_fb": args.lumi,
        "sample": args.sample
    }

    with open(args.output, "wb") as f:
        pickle.dump(out_dict, f)
    print(f"Success! Saved to {args.output}")

if __name__ == "__main__":
    main()