# -*- coding: utf-8 -*-
"""
GridCast - Hydraulic Environment & Containment Controls
Defines the 16-channel fluid matrix and enforces hydraulic equilibrium.
All comments and logs are strictly formatted for industrial monitoring.
"""
import logging
import torch

# Configure the local logger for execution tracking
logger = logging.getLogger("gridcast.core.environment")


class GridCastEnvironment:
    """
    Manages hydraulic pressure, dye containment, and boundary drainage.
    Enforces structural invariants across the 4D block tensor.
    """
    def __init__(self, steps: int = 12, width: int = 30, height: int = 30):
        self.T = steps
        self.C = 16  # 1 Carrier Solvent + 9 Dyes + 6 Routing/Structural channels
        self.X = width
        self.Y = height
        
    def check_hydraulic_leak(self, block: torch.Tensor, atol: float = 1e-5) -> bool:
        """
        Verifies strict fluid containment across the processing pipeline.
        
        1. Drain Test: If a Drain Vent (Z=15) is open, all Chemical Dyes (Z=1..9) must 
           be completely flushed to 0.0, and the Carrier Solvent (Z=0) must backfill 
           the node space to 1.0.
        2. Pressure Test: Total fluid volume (Solvent + Dyes) must equal 1.0 per node.
        
        Args:
            block (torch.Tensor): The 4D space-time tensor of shape (T, C, X, Y).
            atol (float): Absolute tolerance for floating-point comparisons.
            
        Returns:
            bool: True if hydraulic invariants are met, False otherwise.
        """
        assert block.shape == (self.T, self.C, self.X, self.Y), f"Invalid structural tensor shape: {block.shape}"
        
        # 1. Drain Vent Isolation Test
        drain_mask = block[:, 15] > 0.9
        if drain_mask.any():
            # Ensure dyes are fully evacuated under active drains
            dyes_in_drains = block[:, 1:10][drain_mask.unsqueeze(1).expand(-1, 9, -1, -1)]
            if not torch.allclose(dyes_in_drains, torch.tensor(0.0, device=block.device), atol=atol):
                logger.error("Hydraulic Failure: Chemical dye detected inside active drain vents (Z=15).")
                return False
                
            # Ensure Carrier Solvent fills the vent volume
            solvent_in_drains = block[:, 0][drain_mask]
            if not torch.allclose(solvent_in_drains, torch.tensor(1.0, device=block.device), atol=atol):
                logger.error("Hydraulic Failure: Carrier solvent missing from drain relief zone (Z=0 at Z=15).")
                return False

        # 2. Total System Pressure Equalization (Solvent + Dyes == 1.0)
        total_fluid_pressure = block[:, 0:10].sum(dim=1)
        target_pressure = torch.ones_like(total_fluid_pressure)
        
        if not torch.allclose(total_fluid_pressure, target_pressure, atol=atol):
            max_drift = torch.max(torch.abs(total_fluid_pressure - target_pressure)).item()
            logger.error("Hydraulic Failure: System pressure drift detected. Max error: %e", max_drift)
            return False
            
        logger.debug("Hydraulic containment verification passed. Integrity: 100%%.")
        return True

    def enforce_containment(self, block: torch.Tensor) -> torch.Tensor:
        """
        Hardens the tensor by forcibly sealing any numerical drift or float evaporation.
        Projective normalization layer applied to fluid channels (Z=0..9).
        
        Args:
            block (torch.Tensor): The raw 4D tensor before validation.
            
        Returns:
            torch.Tensor: The sealed tensor satisfying all conservation invariants.
        """
        # Step 1: Active Drain Vent processing
        drain_mask = block[:, 15] > 0.9
        if drain_mask.any():
            logger.debug("Active drain venting triggered. Flushing chemical dyes.")
            # Force dyes to 0 where drain vents are open
            for c in range(1, 10):
                block[:, c] = torch.where(drain_mask, torch.tensor(0.0, device=block.device), block[:, c])
            # Force carrier solvent to 1 to backfill the volume
            block[:, 0] = torch.where(drain_mask, torch.tensor(1.0, device=block.device), block[:, 0])

        # Step 2: L1 normalization over Z=0..9 to clear floating-point numerical drift
        block[:, 0:10] = torch.clamp(block[:, 0:10], min=0.0)
        pressure_sum = block[:, 0:10].sum(dim=1, keepdim=True)
        
        # Guard against zero division
        pressure_sum = torch.where(pressure_sum == 0, torch.ones_like(pressure_sum), pressure_sum)
        
        # Rescale fluid channels to stabilize total system pressure
        block[:, 0:10] = block[:, 0:10] / pressure_sum
        
        return block