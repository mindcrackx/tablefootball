#!/usr/bin/env python3

import sys
import itertools

def normalize_encounter(team1, team2):
    """Normalize encounters so AB vs CD = CD vs AB (same matchup, different sides)"""
    return (team1, team2) if team1 <= team2 else (team2, team1)

def get_player_position(player, team1, team2):
    """Returns (team_index, position) where team_index: 0=team1, 1=team2; position: 0=Defense, 1=Attack"""
    if player in team1:
        return (0, team1.index(player))
    else:
        return (1, team2.index(player))

def score_round(candidate, previous_rounds, allow_mirrors=False):
    """Score rounds to maximize position changes and space out mirror encounters"""
    team1, team2 = candidate
    normalized_candidate = normalize_encounter(team1, team2)

    # Prevent duplicate normalized encounters in first 12 rounds
    if not allow_mirrors:
        for prev_team1, prev_team2 in previous_rounds:
            if normalize_encounter(prev_team1, prev_team2) == normalized_candidate:
                return 0

    if not previous_rounds:
        return 100

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
        # Bonus for actual team composition change (2 team changes = different composition)
        # 0 or 4 team changes = same composition, just swapped sides
        if team_changes == 2:
            score += 0.5
    else:
        score = position_changes * 8 + team_changes * 5

    # Space out mirror encounters to avoid repetition feeling too soon
    if allow_mirrors:
        rounds_since_last = None
        for i in range(len(previous_rounds) - 1, -1, -1):
            prev_team1, prev_team2 = previous_rounds[i]
            if normalize_encounter(prev_team1, prev_team2) == normalized_candidate:
                rounds_since_last = len(previous_rounds) - i
                break

        if rounds_since_last is not None:
            # Heavy penalties for recent mirrors to improve variety perception
            if rounds_since_last <= 2:
                score -= 50
            elif rounds_since_last <= 4:
                score -= 25
            elif rounds_since_last <= 6:
                score -= 10

    return score

def generate_all_rounds():
    """Generate all 24 possible combinations of 4 players in 2 teams with 2 positions each"""
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
    """Generate optimal schedule using ranking approach for both halves"""
    all_rounds = generate_all_rounds()
    schedule = []

    # First 12 rounds: unique normalized encounters only
    for _ in range(min(num_rounds, 12)):
        best_score = -1
        best_round = None

        for candidate in all_rounds:
            score = score_round(candidate, schedule, allow_mirrors=False)
            if score > best_score:
                best_score = score
                best_round = candidate

        if best_round is None:
            break

        schedule.append(best_round)

    # Rounds 13-24: allow mirror encounters but space them out intelligently
    # This is much better than naive team swapping which gave poor transitions
    if num_rounds > 12:
        remaining_rounds = num_rounds - 12
        for _ in range(remaining_rounds):
            best_score = -1
            best_round = None

            for candidate in all_rounds:
                if candidate in schedule:
                    continue

                score = score_round(candidate, schedule, allow_mirrors=True)
                if score > best_score:
                    best_score = score
                    best_round = candidate

            if best_round is None:
                break

            schedule.append(best_round)

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

    print("round,team1,team2,sitting")
    for i, (team1, team2) in enumerate(schedule, 1):
        print(f"{i},{team1},{team2},")

if __name__ == "__main__":
    main()