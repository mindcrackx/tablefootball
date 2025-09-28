# Table Football Tournament Scheduler

An optimal constraint programming solution for scheduling table football tournaments with 4 or more players with only one table, maximizing team variety and position switching.


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
python main.py --rounds 24 --players 4

```


### Generate tournament files
```bash
mkdir -p tournaments
python main4.py 24 > ./tournaments/4p.csv
python main5.py 120 > ./tournaments/5p.csv
python main.py --rounds 360 --players 6 > ./tournaments/6p.csv
python main.py --rounds 840 --players 7 > ./tournaments/7p.csv
```
