"""
Generate and save quantum circuit diagrams for QuantumMusic README and docs.
Run from project root: python scripts/generate_circuits.py
Saves images to assets/images/
"""

import os
import sys
import numpy as np

# Add project root for imports if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "assets", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    try:
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.circuit.library import Initialize
    except ImportError as e:
        print("Install qiskit and qiskit[visualization]: pip install qiskit qiskit[visualization]")
        raise e

    np.random.seed(42)

    # ----- 1. Amplitude Encoding Circuit (16D -> 4 qubits) -----
    vector_16 = np.random.randn(16).astype(np.float64)
    vector_16 = vector_16 / np.linalg.norm(vector_16)

    qc_amp = QuantumCircuit(4, name="encoded_state")
    init_gate = Initialize(vector_16)
    qc_amp.append(init_gate, range(4))

    try:
        fig = qc_amp.draw("mpl", style="iqp", fold=-1)
        path = os.path.join(OUTPUT_DIR, "amplitude_encoding_circuit.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        import matplotlib
        matplotlib.pyplot.close(fig)
    except Exception as e:
        print(f"Amplitude encoding (mpl) failed: {e}. Saving text draw.")
        with open(os.path.join(OUTPUT_DIR, "amplitude_encoding_circuit.txt"), "w") as f:
            f.write(qc_amp.draw("text", fold=-1))

    # ----- 2. SWAP Test Circuit -----
    ancilla = QuantumRegister(1, "ancilla")
    reg1 = QuantumRegister(4, "state1")
    reg2 = QuantumRegister(4, "state2")
    c = ClassicalRegister(1, "measure")

    state1 = vector_16
    state2 = np.roll(vector_16, 2)  # different state
    state2 = state2 / np.linalg.norm(state2)

    qc_swap = QuantumCircuit(ancilla, reg1, reg2, c)
    qc_swap.append(Initialize(state1), reg1)
    qc_swap.append(Initialize(state2), reg2)
    qc_swap.h(ancilla[0])
    for i in range(4):
        qc_swap.cswap(ancilla[0], reg1[i], reg2[i])
    qc_swap.h(ancilla[0])
    qc_swap.measure(ancilla[0], c[0])

    try:
        fig = qc_swap.draw("mpl", style="iqp", fold=-1)
        path = os.path.join(OUTPUT_DIR, "swap_test_circuit.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        import matplotlib
        matplotlib.pyplot.close(fig)
    except Exception as e:
        print(f"SWAP test (mpl) failed: {e}. Saving text draw.")
        with open(os.path.join(OUTPUT_DIR, "swap_test_circuit.txt"), "w") as f:
            f.write(qc_swap.draw("text", fold=-1))

    # ----- 3. Grover Oracle (mark one state) -----
    n_qubits = 4
    marked_indices = [3]  # mark |0011⟩

    oracle = QuantumCircuit(n_qubits)
    for idx in marked_indices:
        bin_str = format(idx, f"0{n_qubits}b")
        for qubit, bit in enumerate(bin_str):
            if bit == "0":
                oracle.x(qubit)
        if n_qubits == 1:
            oracle.z(0)
        else:
            oracle.h(n_qubits - 1)
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oracle.h(n_qubits - 1)
        for qubit, bit in enumerate(bin_str):
            if bit == "0":
                oracle.x(qubit)

    try:
        fig = oracle.draw("mpl", style="iqp", fold=-1)
        path = os.path.join(OUTPUT_DIR, "grover_oracle.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        import matplotlib
        matplotlib.pyplot.close(fig)
    except Exception as e:
        with open(os.path.join(OUTPUT_DIR, "grover_oracle.txt"), "w") as f:
            f.write(oracle.draw("text", fold=-1))

    # ----- 4. Grover Diffusion -----
    diffusion = QuantumCircuit(n_qubits)
    diffusion.h(range(n_qubits))
    diffusion.x(range(n_qubits))
    if n_qubits == 1:
        diffusion.z(0)
    else:
        diffusion.h(n_qubits - 1)
        diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        diffusion.h(n_qubits - 1)
    diffusion.x(range(n_qubits))
    diffusion.h(range(n_qubits))

    try:
        fig = diffusion.draw("mpl", style="iqp", fold=-1)
        path = os.path.join(OUTPUT_DIR, "grover_diffusion.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        import matplotlib
        matplotlib.pyplot.close(fig)
    except Exception as e:
        with open(os.path.join(OUTPUT_DIR, "grover_diffusion.txt"), "w") as f:
            f.write(diffusion.draw("text", fold=-1))

    # ----- 5. Full Grover Circuit (1 iteration) -----
    qc_grover = QuantumCircuit(n_qubits)
    qc_grover.h(range(n_qubits))
    qc_grover.append(oracle.to_gate(), range(n_qubits))
    qc_grover.append(diffusion.to_gate(), range(n_qubits))
    qc_grover.measure_all()

    try:
        fig = qc_grover.draw("mpl", style="iqp", fold=-1)
        path = os.path.join(OUTPUT_DIR, "grover_full_circuit.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        import matplotlib
        matplotlib.pyplot.close(fig)
    except Exception as e:
        with open(os.path.join(OUTPUT_DIR, "grover_full_circuit.txt"), "w") as f:
            f.write(qc_grover.draw("text", fold=-1))

    print(f"\nAll circuit outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
