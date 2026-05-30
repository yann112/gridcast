# Physics-Aware ARC Puzzle Solver Environment

## Overview
A continuous physics simulation environment designed to transform ARC puzzles from discrete input-output mappings into continuous trajectory learning problems. Instead of learning direct input→output transformations, this environment models the underlying physical processes that govern puzzle transformations, enabling neural networks to learn the "physics" of puzzle-solving rather than memorizing pattern mappings.

## Key Features
- **Anthropic Temporal Buffer**: Flexible T-dimension accommodates operations of varying durations
- **Physics-Based Dynamics**: Realistic transfer mechanics with intermediate states, elastic timing, and containment rules  
- **Trajectory Learning**: Generates complete transformation sequences for continuous learning
- **Masked Reconstruction**: Trains models to reconstruct hidden physics dynamics from partial observations

## Architecture
- **State Space**: 16-channel grid with specialized layers for content (0-9), walls (10), transfer mechanics (13), commands (14), and elastic timing (15)
- **Action Space**: Drain commands that trigger controlled transfers between layers
- **Physics Engine**: Elastic-timed, half-state transition mechanics with wall containment
- **Training Paradigm**: Full trajectory generation with selective masking for reconstruction learning

## Continuous Physics Design
The environment employs **continuous state representations** crucial for each operation:

- **Intermediate States (0.5)**: Enables gradual transitions rather than instantaneous changes
- **Progress Tracking**: Each cell can be at different phases (0.0→0.5→1.0) independently  
- **Spatial Heterogeneity**: Allows local variation in operation speeds across the grid
- **Neural Control**: Enables CNNs to learn smooth modulation of physical parameters
- **Future Extensibility**: Foundation for curved/spatially-varying physics operations

This continuous approach ensures that operations can adapt their behavior locally while maintaining global consistency, providing the neural network with rich, gradient-friendly physics representations.

## Usage
Designed for generating training datasets where models learn to predict complete transformation physics from partial observations, potentially leading to more robust generalization on novel puzzle configurations.

## Visualization
Integrated plotting tools to inspect:
- Raw input/output grids with actual colors
- Encoded tensors in grayscale (black=0, grey=0.5, white=1)  
- Individual physics channels with hover details (value, layer name, coordinates)
- All 16 channels simultaneously for debugging physics mechanics