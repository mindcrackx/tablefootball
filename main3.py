#!/usr/bin/env python3

import sys
import itertools
from typing import List, Tuple, Dict

def normalize_encounter(team1: str, team2: str) -> Tuple[str, str]:
    """Normalize encounter by ensuring lexicographically smaller team comes first"""
    if team1 <= team2:
        return (team1, team2)
    else:
        return (team2, team1)

def get_player_info(player: str, team1: str, team2: str) -> Tuple[str, str]:
    """Get team and position for a player in a round"""
    if player in team1:
        position = "Defense" if team1.index(player) == 0 else "Attack"
        return ("team1", position)
    elif player in team2:
        position = "Defense" if team2.index(player) == 0 else "Attack"
        return ("team2", position)
    else:
        raise ValueError(f"Player {player} not found in either team")

def generate_all_possible_rounds() -> List[Tuple[str, str]]:
    """Generate all possible team combinations and position arrangements"""
    players = ['A', 'B', 'C', 'D']
    rounds = []

    # Generate all ways to split 4 players into 2 teams of 2
    for team1_players in itertools.combinations(players, 2):
        team2_players = tuple(p for p in players if p not in team1_players)

        # Generate all position arrangements for each team combination
        for team1_perm in itertools.permutations(team1_players):
            for team2_perm in itertools.permutations(team2_players):
                team1 = ''.join(team1_perm)
                team2 = ''.join(team2_perm)
                rounds.append((team1, team2))

    return rounds

def score_round(candidate: Tuple[str, str], previous_rounds: List[Tuple[str, str]]) -> int:
    """Score a candidate round based on how different it is from previous rounds"""
    team1, team2 = candidate
    normalized_candidate = normalize_encounter(team1, team2)

    # Check if this normalized encounter already exists
    for prev_team1, prev_team2 in previous_rounds:
        if normalize_encounter(prev_team1, prev_team2) == normalized_candidate:
            return 0  # Already seen this encounter

    if not previous_rounds:
        return 100  # First round gets full score

    score = 0
    last_team1, last_team2 = previous_rounds[-1]

    # Count different types of changes
    position_changes = 0
    team_changes = 0
    both_changes = 0

    # Score based on position and team changes for each player
    for player in ['A', 'B', 'C', 'D']:
        try:
            last_team, last_pos = get_player_info(player, last_team1, last_team2)
            curr_team, curr_pos = get_player_info(player, team1, team2)

            if curr_pos != last_pos and curr_team != last_team:
                both_changes += 1
            elif curr_pos != last_pos:
                position_changes += 1
            elif curr_team != last_team:
                team_changes += 1

        except ValueError:
            # Player not found - shouldn't happen with 4 players
            continue

    # Prioritize scenarios where ALL players change in some way
    # Highest priority: ALL players switch positions (regardless of teams)
    if position_changes + both_changes == 4:
        score += 100  # All players switch positions

        # Tiebreaker based on actual team composition changes
        # After normalization: 0 or 4 team changes = same composition (just swapped sides)
        # 2 team changes = different team composition
        total_team_changes = team_changes + both_changes
        if total_team_changes == 2:
            score += 0.5  # Bonus for actual team composition change
        # 0 or 4 team changes get no bonus (same composition, just swapped)

    # Second priority: ALL players change teams AND positions
    elif both_changes == 4:
        score += 90   # All players change both

    # Third priority: ALL players change teams only
    elif team_changes + both_changes == 4:
        score += 80   # All players change teams

    # Otherwise, score based on individual changes
    else:
        score += both_changes * 10 + position_changes * 8 + team_changes * 5

    return score

def generate_tournament_schedule(num_rounds: int) -> List[Tuple[str, str]]:
    """Generate tournament schedule using ranking approach"""
    all_possible_rounds = generate_all_possible_rounds()
    schedule = []

    print(f"Generating {num_rounds} rounds using ranking approach...", file=sys.stderr)
    print(f"Total possible rounds: {len(all_possible_rounds)}", file=sys.stderr)

    # Generate unique rounds (max 12 for 4 players)
    unique_rounds_needed = min(num_rounds, 12)

    for round_num in range(unique_rounds_needed):
        best_score = -1
        best_round = None
        tied_rounds = []

        # Score all candidate rounds
        for candidate in all_possible_rounds:
            score = score_round(candidate, schedule)
            if score > best_score:
                best_score = score
                best_round = candidate
                tied_rounds = [candidate]
            elif score == best_score and score > 0:
                tied_rounds.append(candidate)

        if best_round is None:
            print(f"No valid round found for round {round_num + 1}", file=sys.stderr)
            break

        # If there are ties, show them for debugging
        if len(tied_rounds) > 1 and round_num < 5:  # Only show for first few rounds
            print(f"Round {round_num + 1} tied candidates (score {best_score}):", file=sys.stderr)
            for tied_round in tied_rounds[:5]:  # Show first 5 ties
                print(f"  {tied_round[0]} vs {tied_round[1]}", file=sys.stderr)

        schedule.append(best_round)
        team1, team2 = best_round
        normalized = normalize_encounter(team1, team2)
        print(f"Round {round_num + 1}: {team1} vs {team2} (normalized: {normalized[0]} vs {normalized[1]}) - Score: {best_score}", file=sys.stderr)

    # If we need more than 12 rounds, repeat by swapping teams
    if num_rounds > 12:
        print(f"Need {num_rounds - 12} more rounds, repeating with swapped teams...", file=sys.stderr)
        for i in range(num_rounds - 12):
            original_round = schedule[i % 12]
            swapped_round = (original_round[1], original_round[0])  # Swap team1 and team2
            schedule.append(swapped_round)

    return schedule

def print_csv_output(schedule: List[Tuple[str, str]]):
    """Print schedule in CSV format to stdout"""
    print("round,team1,team2,sitting")
    for i, (team1, team2) in enumerate(schedule, 1):
        print(f"{i},{team1},{team2},")

def main():
    if len(sys.argv) != 2:
        print("Usage: python main3.py <num_rounds>", file=sys.stderr)
        sys.exit(1)

    try:
        num_rounds = int(sys.argv[1])
        if num_rounds < 1:
            raise ValueError("Number of rounds must be positive")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Table Football Tournament Scheduler - Ranking Approach", file=sys.stderr)
    print(f"Players: A, B, C, D", file=sys.stderr)
    print(f"Requested rounds: {num_rounds}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    schedule = generate_tournament_schedule(num_rounds)

    print("=" * 50, file=sys.stderr)
    print(f"Generated {len(schedule)} rounds", file=sys.stderr)

    # Print analysis
    unique_encounters = set()
    for team1, team2 in schedule:
        normalized = normalize_encounter(team1, team2)
        unique_encounters.add(normalized)

    print(f"Unique encounters: {len(unique_encounters)}", file=sys.stderr)
    print("Unique encounters:", file=sys.stderr)
    for encounter in sorted(unique_encounters):
        print(f"  {encounter[0]} vs {encounter[1]}", file=sys.stderr)

    # Output CSV to stdout
    print_csv_output(schedule)

if __name__ == "__main__":
    main()