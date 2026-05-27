cat << 'EOF' > README.md
# GridCast 🏭

Continuous hydro-dynamic framework for solving the ARC-Prize 2026 challenge.
A static 4D block-universe system ($12 \times 16 \times 30 \times 30$) driven by strict fluid tight-containment laws and pressure gradient pathfinding.

## Factory Channels (Z=0 to Z=15)
* `Z=0`     : **Base Solvent** (Neutral carrier fluid, hydraulic buffer for volume equalization)
* `Z=1..9`  : **Chemical Dyes** (The 9 visible ARC colors represented as continuous reactants/floats)
* `Z=10`    : **Structural Pipes** (Immutable layout boundaries / passive solid walls)
* `Z=11`    : **Valve Actuators** (Logic layer containing execution routines and operation triggers)
* `Z=12`    : **Zone Gaskets** (Surgical local masking used to seal off or isolate specific shapes)
* `Z=13`    : **Bypass Conduit** (Underground auxiliary layer for non-destructive transport and 90° matrix rotations)
* `Z=14`    : **Pressure Gradient** (Spatio-temporal coordinate field dictating fluid velocity and automatic obstacle steering)
* `Z=15`    : **Drain Vents** (Active exhaust fields that instantly flush incoming dyes and trigger solvent backfill)

## Core Hydraulic Invariants
1. **Hydraulic Equilibrium**: System volume is strictly sealed. Every individual coordinate node must satisfy the pressure constraint $\sum Z_{0..9} = 1.0$ at any given entropy step. Dyes cannot expand without displacing the Base Solvent.
2. **Toroidal Circuit (Donut Flow)**: Pipe layouts form a closed, looping circuit. Grid edges are natively connected along both axes using seamless circular shifts (`torch.roll`), allowing fluid to pass through a boundary and reappear on the opposite side.
3. **Dynamic Venting**: Active `Drain Vents` (`Z=15`) act as open pressure relief zones. Any `Chemical Dye` pushed into these zones is instantly flushed out to `0.0`, with the `Base Solvent` (`Z=0`) vacuum-filling the void to maintain circuit stability.

## Project Architecture
* `gridcast/core/environment.py`: Monitors and stabilizes hydraulic integrity, mitigating floating-point drift via L1 projective normalization.
* `gridcast/core/generator.py`: Contains pure matrix operators for hydraulic displacement, sédimentation, and closed-loop routing.
* `gridcast/core/phase_factory.py`: Generates the continuous vector fields and slopes for the `Pressure Gradient` channel.
* `gridcast/models/model.py`: Frugal `Conv3D` network designed to map current routing states to execution steps by following gradient lines.
EOF