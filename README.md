# Table Football Tournament Scheduler

An optimal constraint programming solution for scheduling table football tournaments with 4 players, maximizing team variety and position switching.

## Features

- **Optimal Scheduling**: Uses CPMpy (Constraint Programming) to find mathematically optimal tournament schedules
- **Position Switching**: Prioritizes alternating players between Defense and Attack positions
- **Team Mixing**: Maximizes variety in team compositions across rounds
- **Scalable**: Handles tournaments from 1-24 rounds (maximum possible unique encounters)
- **Fast Performance**: Generates optimal 24-round schedules in <400ms

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Generate optimal 24-round tournament
python main.py

# Run examples with different tournament lengths
python examples.py

# Run test suite
python -m pytest test_scheduler.py -v
```

## How It Works

### Tournament Rules
- **4 players** (A, B, C, D)
- **2 teams of 2** players each round
- **Defense and Attack** positions per team
- **Team notation**: First letter = Defense, Second letter = Attack

### Optimization Goals
1. **No exact repetitions**: Each encounter is unique
2. **Maximum team mixing**: Players change teammates frequently
3. **Position switching**: Players alternate between Defense and Attack
4. **Standardized start**: Round 1 always begins with AB vs CD

### Example Output
```
Round 1: AB vs CD  // A=Defense, B=Attack vs C=Defense, D=Attack
Round 2: BC vs DA  // All 4 players switch positions!
Round 3: AD vs CB  // Maximum team variety
...
```

## Technical Details

- **Algorithm**: Constraint Satisfaction Problem (CSP) using CPMpy
- **Solver**: OR-Tools CP-SAT (Google's constraint solver)
- **Optimization**: Multi-objective scoring with weighted priorities
- **Performance**: O(n) scaling for practical tournament lengths

## Files

- `main.py` - Core scheduler implementation
- `test_scheduler.py` - Comprehensive test suite (13 tests)
- `examples.py` - Usage examples and demonstrations
- `PROBLEM.md` - Detailed problem specification
- `requirements.txt` - Python dependencies

## License

Open source - feel free to adapt for your tournaments!