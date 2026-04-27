import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. Damped Pendulum Data Generation
# ==========================================
def damped_pendulum(t, y, b, c):
    """
    Nonlinear ODE for a damped pendulum.
    b: damping coefficient
    c: gravity/length parameter
    """
    theta, omega = y
    dydt = [omega, -b * omega - c * np.sin(theta)]
    return dydt

def generate_pendulum_data(num_trajectories=200, t_span=(0, 10), dt=0.1):
    """
    Generates training data containing state transitions x_k -> x_{k+1} 
    from simulated damped pendulum trajectories.
    """
    b = 0.15
    c = 1.0  
    
    t_eval = np.arange(t_span[0], t_span[1], dt)
    dataset_X = []
    dataset_Y = []
    
    for _ in range(num_trajectories):
        # Random initial conditions: theta in [-pi, pi], omega in [-1, 1]
        theta0 = np.random.uniform(-np.pi, np.pi)
        omega0 = np.random.uniform(-1, 1)
        y0 = [theta0, omega0]
        
        # Simulate trajectory
        sol = solve_ivp(damped_pendulum, t_span, y0, t_eval=t_eval, args=(b, c))
        states = sol.y.T  # Shape: (num_steps, 2)
        
        # Inputs: x_k, Outputs: x_{k+1}
        X = states[:-1, :]
        Y = states[1:, :]
        
        dataset_X.append(X)
        dataset_Y.append(Y)
        
    return np.vstack(dataset_X), np.vstack(dataset_Y)

# ==========================================
# 2. Structured Neural Network (StNN) Module
# ==========================================
class StNN(nn.Module):
    def __init__(self, state_dim, p=4):
        """
        Implementation of the StNN based on the Hankel operator factorization.
        state_dim (n): Dimension of the system states.
        p: Number of parallel sub-weight blocks.
        """
        super().__init__()
        self.n = state_dim
        self.r = 2 * self.n
        self.p = p
        
        # Random diagonal matrices required by the factorization
        self.D_hat = nn.Parameter(torch.randn(self.r))
        self.D_grave = nn.Parameter(torch.randn(self.n))
        self.D_check = nn.Parameter(torch.randn(self.n))
        
        # Learnable symmetric sub-weight approximation of scaled DFT matrix
        self.F_n = nn.Parameter(torch.randn(self.n, self.n) / np.sqrt(self.n))
        
        # Biases for each layer
        self.b1 = nn.Parameter(torch.zeros(self.p * self.r))
        self.b2 = nn.Parameter(torch.zeros(self.p * self.n))
        self.b3 = nn.Parameter(torch.zeros(self.p * self.n))
        self.b4 = nn.Parameter(torch.zeros(self.n))
        
        # Frozen components: Identity and Anti-diagonal Identity matrices
        self.register_buffer('I_n', torch.eye(self.n))
        self.register_buffer('I_anti', torch.fliplr(torch.eye(self.n)))

    def forward(self, x):
        device = x.device
        
        # Reconstruct H_r = [I_n, I_n; D_grave, -D_grave]
        Hr_top = torch.cat([self.I_n, self.I_n], dim=1)
        D_g = torch.diag(self.D_grave)
        Hr_bot = torch.cat([D_g, -D_g], dim=1)
        Hr = torch.cat([Hr_top, Hr_bot], dim=0) # (2n x 2n)
        
        # Reconstruct block diagonal F_n
        Fn_blk = torch.block_diag(self.F_n, self.F_n) # (2n x 2n)
        
        # Permutation matrix P_r (interleaves even-odd)
        idx_even = torch.arange(0, self.r, 2, device=device)
        idx_odd = torch.arange(1, self.r, 2, device=device)
        idx = torch.cat([idx_even, idx_odd])
        P_r = torch.eye(self.r, device=device)[idx]
        
        # Reconstruct structured matrix F_r
        Fr = P_r.T @ Fn_blk @ Hr
        
        # J matrix for zero-padding = [I_n; 0_n]
        J = torch.cat([self.I_n, torch.zeros(self.n, self.n, device=device)], dim=0)
        
        # === Layer 1 ===
        # w_{1,0} = F_r @ J
        w1_0 = Fr @ J
        W1_0 = w1_0.repeat(self.p, 1) # Parallel blocks formulation
        h1 = torch.tanh(x @ W1_0.T + self.b1)
        
        # === Layer 2 ===
        # w_{2,1} = J^T @ F_r @ diag(D_hat)
        w2_1 = J.T @ Fr @ torch.diag(self.D_hat)
        h1_reshaped = h1.view(-1, self.r)
        h2 = h1_reshaped @ w2_1.T
        h2 = h2.view(-1, self.p * self.n) + self.b2
        h2 = torch.sigmoid(h2)
        
        # === Layer 3 ===
        # w_{3,2} = Antidiagonal Identity
        h2_reshaped = h2.view(-1, self.n)
        h3 = h2_reshaped @ self.I_anti.T
        h3 = h3.view(-1, self.p * self.n) + self.b3
        h3 = torch.relu(h3)
        
        # === Output Layer ===
        # w_{4,3} = diag(D_check)
        w4_3 = torch.diag(self.D_check)
        h3_reshaped = h3.view(-1, self.p, self.n)
        out = h3_reshaped @ w4_3.T
        out = out.sum(dim=1) + self.b4 # Sums the parallel features
        
        return out

import os
import matplotlib.pyplot as plt

def plot_validation_rollouts(model, num_samples=3, t_span=(0, 10), dt=0.1, out_dir="output"):
    """
    Evaluates the model on new initial conditions and rolls out the 
    trajectory autoregressively, comparing it to the true ODE solution.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    t_eval = np.arange(t_span[0], t_span[1], dt)
    b, c = 0.15, 1.0 # Parameters from damped_pendulum
    
    print(f"Generating and saving {num_samples} validation plots in '{out_dir}/'...")
    for i in range(num_samples):
        # 1. Generate Ground Truth Trajectory
        theta0 = np.random.uniform(-np.pi, np.pi)
        omega0 = np.random.uniform(-1, 1)
        y0 = [theta0, omega0]
        
        sol = solve_ivp(damped_pendulum, t_span, y0, t_eval=t_eval, args=(b, c))
        true_states = sol.y.T # (num_steps, 2)
        
        # 2. Autoregressive Rollout with StNN
        pred_states = np.zeros_like(true_states)
        pred_states[0] = true_states[0] # Provide true initial condition
        
        curr_state = torch.tensor(pred_states[0], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            for k in range(1, len(t_eval)):
                next_state = model(curr_state)
                pred_states[k] = next_state.numpy()[0]
                curr_state = next_state # Feedback prediction as next input
                
        # 3. Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        ax1.plot(t_eval, true_states[:, 0], 'k-', label='True Theta (Angle)')
        ax1.plot(t_eval, pred_states[:, 0], 'r--', label='Predicted Theta')
        ax1.set_ylabel('Theta')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t_eval, true_states[:, 1], 'k-', label='True Omega (Angular Velocity)')
        ax2.plot(t_eval, pred_states[:, 1], 'b--', label='Predicted Omega')
        ax2.set_ylabel('Omega')
        ax2.set_xlabel('Time (s)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Autoregressive Rollout: Validation Sample {i+1}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(out_dir, f'val_rollout_{i+1}.png'), dpi=150)
        plt.close()


# ==========================================
# 3. Training & Validation Loop Setup
# ==========================================
if __name__ == "__main__":
    # Generate Training Data
    X_train_np, Y_train_np = generate_pendulum_data(num_trajectories=10000)
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
    
    # Generate Validation Data (single-step validation)
    X_val_np, Y_val_np = generate_pendulum_data(num_trajectories=40)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    Y_val = torch.tensor(Y_val_np, dtype=torch.float32)
    
    # Initialize StNN Model
    state_dim = X_train.shape[1] # 2 (theta, omega)
    model = StNN(state_dim=state_dim, p=10)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 200
    batch_size = 128
    
    print("Training StNN on Damped Pendulum Dynamics...")
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        # Validation Loop
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, Y_val).item()

            
        avg_train_loss = epoch_loss / X_train.size(0)
            
        if (epoch+1) % 20 == 0:
            plot_validation_rollouts(model, num_samples=5, out_dir="output")
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # After training, test autoregressive rollouts and save plots
    print("\nTraining Complete. Initiating Plotting Phase...")
    plot_validation_rollouts(model, num_samples=5, out_dir="output")
    print("Plots saved successfully!")