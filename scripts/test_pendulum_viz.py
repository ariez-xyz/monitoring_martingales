#!/usr/bin/env python3
"""Test script for the inverted pendulum visualization."""
import torch
from time import sleep
from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

def main():
    print("Loading pendulum adapter...")
    adapter = NeuralCLBFPendulum(vis_every=5, dt=0.001, noise_level=0.1)

    # Set a specific initial state for interesting dynamics
    # theta=1.2 rad (~69 deg), theta_dot=0.8 rad/s
    adapter.reset(initial_state=torch.tensor([1.2, 3]))

    print(f"Initial state: θ={adapter.state[0]:.3f}, θ̇={adapter.state[1]:.3f}")
    print(f"Initial V: {adapter.clf_history[-1]:.4f}")
    print("\nRunning simulation with visualization...")
    print("(Close the plot window or press Ctrl+C to stop)\n")

    try:
        for i in range(5000):
            adapter.step()
            if i % 50 == 0:
                print(f"Step {i}: θ={adapter.state[0]:.3f}, θ̇={adapter.state[1]:.3f}, V={adapter.clf_history[-1]:.4f}")
            if adapter.done():
                print(f"\nGoal reached at step {i}!")
                break
            sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped by user")

    print(f"\nFinal state: θ={adapter.state[0]:.6f}, θ̇={adapter.state[1]:.6f}")
    print(f"Final V: {adapter.clf_history[-1]:.6f}")

    # Keep the plot open
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
