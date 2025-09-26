# Table Football Tournament Scheduling Problem

## Overview

We need to create an optimal scheduling algorithm for table football tournaments with 4-6 players playing 5-15 rounds, maximizing unique encounters while balancing player positions.

## Game Setup

- **Table Football**: 2 teams of 2 players each
- **Positions**: Each team has 2 positions - Attack and Defense
- **Players**: 4, 5, or 6 total players
- **Rounds**: 5-15 games per tournament

## Player Configurations

### 4 Players (A, B, C, D)
- All players play every round
- Example: Round 1: AB vs CD (A=Attack, B=Defense vs C=Attack, D=Defense)

### 5 Players (A, B, C, D, E)
- 4 players play, 1 player sits out each round
- Sitting player rotates between rounds

### 6 Players (A, B, C, D, E, F)
- 4 players play, 2 players sit out each round
- Sitting players rotate between rounds

## Optimization Goals

### Primary Goal: Maximize Unique Encounters
- **Never repeat exact team compositions**: If AB vs CD occurs in round 1, it should never occur again
- **Prioritize different team pairings**: Avoid same teammates playing together repeatedly
- **Vary opponents**: Each player should face different opponents across rounds

### Secondary Goal: Position Rotation (Soft Constraint)
- Each player should alternate between Attack and Defense positions when possible
- Ideal pattern: Attack → Defense → Attack → Defense...

## Constraints

1. **Exact team repetition forbidden**: Same 4 players in identical team compositions cannot occur twice
2. **Position switching preferred**: Players should change positions between rounds when feasible
3. **Equal participation**: In 5-6 player games, sitting players must rotate fairly
4. **Tournament length**: Algorithm must work effectively for 5-15 rounds

## Example Scenarios

### 4 Players - Round Progression
*Note: In team notation, first letter = Defense, second letter = Attack*

```
Round 1: AB vs CD  // A=Defense, B=Attack vs C=Defense, D=Attack
Round 2: CA vs DB  // Different teams + all players switch positions
Round 3: AD vs BC  // Different teams + all players switch positions
```

### 5 Players - Round Progression
```
Round 1: AB vs CD (E sits)
Round 2: BE vs DA (C sits)
Round 3: ED vs AC (B sits)
```

## Success Metrics

1. **Unique encounter count**: Number of distinct team vs team matchups
2. **Position balance**: How evenly players alternate between Attack/Defense
3. **Participation equity**: Fair distribution of playing time (5-6 player scenarios)
4. **Scalability**: Algorithm performance across 5-15 rounds

## Challenge

Design an algorithm that generates optimal schedules balancing maximum unique encounters with position rotation preferences, while ensuring fair participation across tournament lengths of 5-15 rounds.