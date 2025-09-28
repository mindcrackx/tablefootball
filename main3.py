#!/usr/bin/env python3

import sys
import itertools

def normalize_encounter(team1, team2):
    """Normalize encounter by ensuring lexicographically smaller team comes first"""
    return (team1, team2) if team1 <= team2 else (team2, team1)

def get_player_position(player, team1, team2):
    """Get (team_index, position) for a player. team_index: 0=team1, 1=team2"""
    if player in team1:
        return (0, team1.index(player))  # 0=Defense, 1=Attack
    else:
        return (1, team2.index(player))

def score_round(candidate, previous_rounds):
    """Score a candidate round based on how different it is from previous rounds"""
    team1, team2 = candidate

    # Check if this normalized encounter already exists
    normalized_candidate = normalize_encounter(team1, team2)
    for prev_team1, prev_team2 in previous_rounds:
        if normalize_encounter(prev_team1, prev_team2) == normalized_candidate:
            return 0  # Already seen this encounter

    if not previous_rounds:
        return 100  # First round gets full score

    # Compare with last round
    last_team1, last_team2 = previous_rounds[-1]

    position_changes = 0
    team_changes = 0

    for player in ['A', 'B', 'C', 'D']:
        last_team_idx, last_pos = get_player_position(player, last_team1, last_team2)
        curr_team_idx, curr_pos = get_player_position(player, team1, team2)

        if curr_pos != last_pos:
            position_changes += 1
        if curr_team_idx != last_team_idx:
            team_changes += 1

    # Prioritize all players switching positions
    if position_changes == 4:
        score = 100
        # Bonus for actual team composition change (exactly 2 team changes)
        if team_changes == 2:
            score += 0.5
        return score

    # Otherwise, score based on individual changes
    return position_changes * 8 + team_changes * 5

def generate_all_rounds():
    """Generate all possible team combinations and position arrangements"""
    players = ['A', 'B', 'C', 'D']
    rounds = []

    for team1_players in itertools.combinations(players, 2):
        team2_players = [p for p in players if p not in team1_players]

        for team1_perm in itertools.permutations(team1_players):
            for team2_perm in itertools.permutations(team2_players):
                team1 = ''.join(team1_perm)
                team2 = ''.join(team2_perm)
                rounds.append((team1, team2))

    return rounds

def generate_schedule(num_rounds):
    """Generate tournament schedule using ranking approach"""
    all_rounds = generate_all_rounds()
    schedule = []

    # Generate unique rounds (max 12 for 4 players)
    for _ in range(min(num_rounds, 12)):
        best_score = -1
        best_round = None

        for candidate in all_rounds:
            score = score_round(candidate, schedule)
            if score > best_score:
                best_score = score
                best_round = candidate

        if best_round is None:
            break

        schedule.append(best_round)

    # If we need more than 12 rounds, repeat by swapping teams
    if num_rounds > 12:
        for i in range(num_rounds - 12):
            original = schedule[i % 12]
            schedule.append((original[1], original[0]))  # Swap teams

    return schedule

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

    schedule = generate_schedule(num_rounds)

    # Output CSV to stdout
    print("round,team1,team2,sitting")
    for i, (team1, team2) in enumerate(schedule, 1):
        print(f"{i},{team1},{team2},")

if __name__ == "__main__":
    main()