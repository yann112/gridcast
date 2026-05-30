# -*- coding: utf-8 -*-
"""
GridCast - Masked Synthetic Generator
Generates full 4D trajectories and applies spatial-temporal masking matrices 
to train the network on physical infilling.
"""
import random
import torch
from gridcast.core.environment import GridCastEnvironment
from gridcast.core.generator import apply_hydraulic_displacement
from gridcast.core.phase_factory import generate_diagonal_slope


class MaskedHydraulicDataset:
    def __init__(self, steps: int = 12, width: int = 30, height: int = 30):
        self.env = GridCastEnvironment(steps=steps, width=width, height=height)
        
    def generate_masked_sample(self, mask_ratio: float = 0.3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a complete ground truth 4D block and a masked input version.
        
        Returns:
            input_block (torch.Tensor): The universe with hidden (zeroed) zones.
            target_block (torch.Tensor): The full true physics universe.
            mask_matrix (torch.Tensor): Binary tracking matrix (1.0 where masked).
        """
        # 1. Génération de la trajectoire de référence (Véridique 4D)
        target_block = torch.zeros((self.env.T, self.env.C, self.env.X, self.env.Y), dtype=torch.float32)
        target_block[:, 0] = 1.0 # Solvant de base
        
        # Injection de tuyauteries fixes
        if random.random() > 0.4:
            wall_idx = random.randint(5, 25)
            target_block[:, 10, wall_idx, :] = 1.0
            
        # Injection d'un pigment coloré initial
        dye = random.randint(1, 9)
        sx, sy = random.randint(2, 20), random.randint(2, 20)
        target_block[0, dye, sx:sx+3, sy:sy+3] = 1.0
        target_block[0, 0, sx:sx+3, sy:sy+3] = 0.0 # Nettoyage solvant
        
        # Application d'un gradient de pression constant
        dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        angle = 0.0 if dx == 1 else (180.0 if dx == -1 else (90.0 if dy == 1 else 270.0))
        for t in range(self.env.T):
            target_block = generate_diagonal_slope(target_block, t, angle)
            
        # Simulation de l'écoulement
        for t in range(1, self.env.T):
            target_block[t, 0:11] = target_block[t-1, 0:11].clone()
            target_block = apply_hydraulic_displacement(target_block, t=t, dx=dx, dy=dy, shove_solvent=True)
            target_block = self.env.enforce_containment(target_block)

        # 2. Création du masque d'effacement
        # Masque binaire de même dimension que la zone fluide (T, 10, X, Y)
        mask_matrix = torch.zeros((self.env.T, 10, self.env.X, self.env.Y), dtype=torch.float32)
        
        input_block = target_block.clone()
        
        # Choix de la stratégie de masquage pour ce batch
        strategy = random.choice(["temporal", "spatial", "mixed"])
        
        if strategy == "temporal":
            # On coupe les derniers pas de temps (Inférence classique)
            cut_t = random.randint(4, 9)
            input_block[cut_t:, 0:10] = 0.0
            mask_matrix[cut_t:, 0:10] = 1.0
            
        elif strategy == "spatial":
            # On découpe une boîte aveugle au centre de la timeline
            x0, y0 = random.randint(2, 15), random.randint(2, 15)
            input_block[:, 0:10, x0:x0+12, y0:y0+12] = 0.0
            mask_matrix[:, 0:10, x0:x0+12, y0:y0+12] = 1.0
            
        else:
            # Masquage mixte aléatoire par blocs spatio-temporels
            for t in range(self.env.T):
                if random.random() > 0.5:
                    input_block[t, 0:10, :, :] = 0.0
                    mask_matrix[t, 0:10, :, :] = 1.0

        return input_block, target_block, mask_matrix