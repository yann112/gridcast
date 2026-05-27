# -*- coding: utf-8 -*-
"""
GridCast - Training Pipeline
Triple Loss: Reconstruction (MSE), Pressure Conservation, and Surface Tension (TV Loss).
Tracks execution using structured logging and exports diagnostic fluid timelines.
"""
import logging
import os
import torch
import torch.optim as optim
import torch.nn as nn

from gridcast.models.model import HydraulicRoutingNetwork
from gridcast.utils.synthetic_generator import SyntheticHydraulicDataset
from gridcast.core.environment import GridCastEnvironment
from gridcast.utils.flow_visualizer import plot_hydraulic_trajectory

# Configure industrial logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("gridcast.training")


def loss_cohesion_couleur(block: torch.Tensor) -> torch.Tensor:
    """
    Total Variation (TV) Loss applied per chemical dye channel (Z=1..9).
    Mimics mercury-like surface tension, preventing fluid dispersion
    and forcing sharp, cohesive ARC boundaries.
    """
    # block shape: (Batch, Channels, Time, X, Y)
    diff_h = block[:, 1:10, :, :, 1:] - block[:, 1:10, :, :, :-1]
    diff_v = block[:, 1:10, :, 1:, :] - block[:, 1:10, :, :-1, :]
    return torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_v))


def loss_pressure_conservation(block: torch.Tensor) -> torch.Tensor:
    """
    Penalizes deviations from total system pressure conservation (Sum Z_0..9 == 1.0).
    Ensures that fluid cannot evaporate or multiply across space-time steps.
    """
    total_pressure = block[:, 0:10, :, :, :].sum(dim=1)  # Sum solvent + dyes
    target_pressure = torch.ones_like(total_pressure)
    return nn.functional.mse_loss(total_pressure, target_pressure)


def train(epochs: int = 250, steps_per_epoch: int = 32, lr: float = 1e-3):
    logger.info("🚀 Démarrage de la pompe d'entraînement GridCast...")
    
    # 1. Hardware acceleration check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Executing pipeline on hardware unit: %s", device)
    
    # 2. Instantiate components
    dataset = SyntheticHydraulicDataset(steps=12, width=30, height=30)
    env = GridCastEnvironment(steps=12, width=30, height=30)
    model = HydraulicRoutingNetwork(in_channels=16, hidden_dim=32).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_reconstruction = nn.MSELoss()
    
    # Create diagnostics output folder
    os.makedirs("diagnostics", exist_ok=True)
    
    # 3. Main Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_recon_loss = 0.0
        epoch_press_loss = 0.0
        epoch_tv_loss = 0.0
        epoch_total_loss = 0.0
        
        for _ in range(steps_per_epoch):
            # Generate a fresh synthetic physics trajectory
            input_b, target_b = dataset.generate_single_trajectory()
            
            # Conv3D expects input shape: (Batch, Channels, Time, X, Y)
            # Our dataset outputs standard circuit layout: (Time, Channels, X, Y)
            input_tensor = input_b.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            target_tensor = target_b.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            
            # Forward execution through neural plumbing
            predictions = model(input_tensor)
            
            # Calculate components of the Triple Loss
            loss_recon = criterion_reconstruction(predictions[:, 0:10], target_tensor[:, 0:10])
            loss_press = loss_pressure_conservation(predictions)
            loss_tv = loss_cohesion_couleur(predictions)
            
            # Total Loss Balancing Coefficients
            # Give high priority to reconstruction and containment stability
            total_loss = loss_recon + 0.5 * loss_press + 0.05 * loss_tv
            
            # Backward pass & optimization step
            total_loss.backward()
            optimizer.step()
            
            # Accumulate statistics
            epoch_recon_loss += loss_recon.item()
            epoch_press_loss += loss_press.item()
            epoch_tv_loss += loss_tv.item()
            epoch_total_loss += total_loss.item()
            
        # Standardize metric metrics over the batch steps
        avg_recon = epoch_recon_loss / steps_per_epoch
        avg_press = epoch_press_loss / steps_per_epoch
        avg_tv = epoch_tv_loss / steps_per_epoch
        avg_total = epoch_total_loss / steps_per_epoch
        
        # Log telemetry at regular milestones
        if epoch == 1 or epoch % 25 == 0:
            logger.info(
                "Epoch %03d/%03d | Total Loss: %.5f [Recon: %.5f, Press: %.5f, TV: %.5f]",
                epoch, epochs, avg_total, avg_recon, avg_press, avg_tv
            )
            
            # Visual Diagnostic Generation
            # Detach the first sample, run it through the environment's hard seal layer, and plot
            model.eval()
            with torch.no_grad():
                sample_pred = predictions[0].permute(1, 0, 2, 3).cpu() # Return back to (T, C, X, Y)
                sealed_sample = env.enforce_containment(sample_pred)
                
                # Plot the model's decoded prediction timeline
                plot_hydraulic_trajectory(
                    sealed_sample, 
                    save_path=f"diagnostics/epoch_{epoch:03d}_prediction.png"
                )
                # Plot the ground-truth target timeline for direct structural comparison
                plot_hydraulic_trajectory(
                    target_b, 
                    save_path=f"diagnostics/epoch_{epoch:03d}_ground_truth.png"
                )
            model.train()
            
    # Save final hardened parameters
    torch.save(model.state_dict(), "hydraulic_routing_network.pt")
    logger.info("💾 Modèle sauvegardé avec succès sous 'hydraulic_routing_network.pt'.")


if __name__ == "__main__":
    train()