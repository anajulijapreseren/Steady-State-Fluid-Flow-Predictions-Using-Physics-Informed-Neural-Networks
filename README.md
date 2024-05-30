# Steady-State Fluid Flow Predictions Using Physics-Informed Neural Networks

This project explores the use of Physics-Informed Neural Networks (PINNs) to solve steady-state fluid flow problems. The project includes both numerical solutions using traditional methods and solutions using PINNs.

## Project Structure

### PINN_CODE
This directory contains code for solving steady-state fluid flow problems using PINNs. Each file in this directory is designed to solve a specific problem, utilize different loss functions, or run on different hardware configurations (CPU/GPU). Note that due to time constraints, I was unable to consolidate all functionalities into a single script, which would be much nicer.

**Recommended script:**
- `main_inlet.py`: This script solves the inlet-outlet flow problem using PINNs.

### PINN_SOLUTIONS
This directory contains the visual results obtained from training the PINNs. To reproduce the models and predictions, you will need to run the training scripts, as only the visual outputs are saved. The training process is relatively fast.

### NUMERICAL_SOLUTIONS
This directory contains code and results for solving the fluid flow problems using traditional numerical methods.

### lib_torch
This directory includes utility scripts for building and training neural networks:
- `advnetwork.py`: Defines a neural network with adjustable parameters.
- `NSGLayer.py`: Utilizes automatic differentiation to compute gradients necessary for the physics-informed loss function.
- `pinn.py`: Combines the physical loss, utilizing the gradients, and the neural network.

