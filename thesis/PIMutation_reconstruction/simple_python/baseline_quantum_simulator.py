import numpy as np

class BaselineQuantumSimulator:
    def __init__(self, num_qubits):
        """
        Initializes the state vector for an n-qubit system.
        The state vector has 2^n elements, initialized to the |00...0> state.
        """
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        # Initialize an array of complex numbers [cite: 132]
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0 + 0.0j
        
        # Define foundational 2x2 quantum gates
        self.I = np.eye(2, dtype=np.complex128)
        self.X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        
        # Define 4x4 CNOT gate
        self.CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)

    def _build_full_unitary(self, gate, target_qubit):
        """
        Constructs the 2^n x 2^n unitary matrix for a single-qubit gate 
        applied to a specific target qubit using the Kronecker product.
        """
        matrices = [self.I] * self.num_qubits
        matrices[target_qubit] = gate
        
        full_matrix = matrices[0]
        for i in range(1, self.num_qubits):
            full_matrix = np.kron(full_matrix, matrices[i])
            
        return full_matrix

    def apply_single_gate(self, gate, target_qubit):
        """
        Applies a single-qubit gate by performing matrix-vector multiplication.
        """
        full_u = self._build_full_unitary(gate, target_qubit)
        # Standard matrix-vector multiplication [cite: 131]
        self.state = np.dot(full_u, self.state)

    def apply_cnot(self, control_qubit, target_qubit):
        """
        A naive CNOT application. For strict baseline simplicity, we assume 
        adjacent qubits (control=0, target=1) if not dynamically generating 
        the sparse swap matrix.
        """
        if abs(control_qubit - target_qubit) != 1:
            raise NotImplementedError("Baseline currently supports adjacent CNOTs for simplicity.")
        
        # If control is before target, use standard CNOT
        gate = self.CNOT if control_qubit < target_qubit else self._reverse_cnot()
        
        # Pad with Identity matrices
        matrices = [self.I] * self.num_qubits
        matrices[min(control_qubit, target_qubit)] = gate
        matrices.pop(max(control_qubit, target_qubit)) # Remove one I since CNOT covers 2 qubits
        
        full_matrix = matrices[0]
        for i in range(1, len(matrices)):
            full_matrix = np.kron(full_matrix, matrices[i])
            
        self.state = np.dot(full_matrix, self.state)

    def _reverse_cnot(self):
        """Helper to flip CNOT control/target."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ], dtype=np.complex128)

    def get_probabilities(self):
        """Calculates measurement probabilities (squared amplitudes)."""
        return np.abs(self.state) ** 2

# --- Scientific Verification ---
if __name__ == "__main__":
    # Create a 2-qubit system
    sim = BaselineQuantumSimulator(num_qubits=2)
    
    # Apply a Hadamard gate to qubit 0 to create a superposition
    sim.apply_single_gate(sim.H, target_qubit=0)
    
    # Apply CNOT to entangle qubit 0 and qubit 1 (creating a Bell state)
    sim.apply_cnot(control_qubit=0, target_qubit=1)
    
    print("Baseline State Vector:")
    print(np.round(sim.state, 4))
    print("\nProbabilities:")
    print(np.round(sim.get_probabilities(), 4))