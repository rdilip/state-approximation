"""
Generic landing place for plots and data analysis.
"""

data_points = [round(i,1) for i in np.linspace(0.1, 2.0, 20)]
fidelities = []
fidelity_SH = []
for data_point in data_points:
    print(data_point)
    with open(f"sh_data/T{data_point}.pkl", "rb") as f:
        Psi = pickle.load(f)
    with open(f"single_layer_approximations/T{data_point}_depth1_circuit.pkl","rb") as f:
        Ulist = pickle.load(f)
    Psi_from_Ulist = generate_state_from_unitary_list(Ulist, Lambda=None, reverse=False)
    fidelity_SH.append(np.linalg.norm(mps_overlap(Psi_from_Ulist, Psi.copy()))**2)
    A, Lambda, F = diagonal_expansion(Psi.copy(), eta=1, num_sweeps=20)
    fidelities.append(np.array(F)**2)
