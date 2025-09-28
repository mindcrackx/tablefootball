#!/usr/bin/env python3

import sys
import itertools

def normalize_encounter(team1, team2):
    """Normalize encounters so AB vs CD = CD vs AB (same matchup, different sides)"""
    return (team1, team2) if team1 <= team2 else (team2, team1)

def get_player_position(player, team1, team2):
    """Returns (team_index, position) where team_index: 0=team1, 1=team2; position: 0=Defense, 1=Attack
    Returns None if player is not in either team (sitting)"""
    if player in team1:
        return (0, team1.index(player))
    elif player in team2:
        return (1, team2.index(player))
    else:
        return None  # Player is sitting

def get_sitting_player(round_num):
    """Get which player sits for a given round number (1-indexed)"""
    # Rotation pattern: E sits round 1, A sits round 2, B sits round 3, C sits round 4, D sits round 5, then repeat
    sitting_order = ['E', 'A', 'B', 'C', 'D']
    return sitting_order[(round_num - 1) % 5]

def score_round(candidate, previous_rounds, allow_mirrors=False):
    """Score rounds to maximize position changes and space out mirror encounters"""
    team1, team2, sitting = candidate
    normalized_candidate = normalize_encounter(team1, team2)

    # For the simple 5-round cycle approach, we only prevent exact duplicates
    # within the current cycle (5 rounds), not normalized duplicates
    if not allow_mirrors:
        # Check last 5 rounds for exact duplicates only
        recent_rounds = previous_rounds[-5:] if len(previous_rounds) >= 5 else previous_rounds
        for prev_team1, prev_team2, prev_sitting in recent_rounds:
            if (team1, team2, sitting) == (prev_team1, prev_team2, prev_sitting):
                return 0

    if not previous_rounds:
        return 100

    last_team1, last_team2, last_sitting = previous_rounds[-1]
    position_changes = 0
    team_changes = 0

    # Only score among the 4 playing players
    playing_players = [p for p in ['A', 'B', 'C', 'D', 'E'] if p != sitting]

    for player in playing_players:
        last_position = get_player_position(player, last_team1, last_team2)
        curr_position = get_player_position(player, team1, team2)

        # Skip comparison if player was sitting in previous round
        if last_position is None or curr_position is None:
            continue

        last_team_idx, last_pos = last_position
        curr_team_idx, curr_pos = curr_position

        if curr_pos != last_pos:
            position_changes += 1
        if curr_team_idx != last_team_idx:
            team_changes += 1

    # Prioritize all playing players switching positions
    if position_changes >= 3:  # More lenient for 5-player case
        score = 100
        # Bonus for actual team composition change (2 team changes = different composition)
        if team_changes == 2:
            score += 0.5
    else:
        score = position_changes * 8 + team_changes * 5

    return score

def generate_all_rounds():
    """Generate all possible combinations for 5 players with 1 sitting"""
    all_players = ['A', 'B', 'C', 'D', 'E']
    rounds = []

    # For each possible sitting player
    for sitting_player in all_players:
        playing_players = [p for p in all_players if p != sitting_player]

        # Generate all ways to split 4 playing players into 2 teams of 2
        for team1_players in itertools.combinations(playing_players, 2):
            team2_players = [p for p in playing_players if p not in team1_players]

            for team1_perm in itertools.permutations(team1_players):
                for team2_perm in itertools.permutations(team2_players):
                    team1 = ''.join(team1_perm)
                    team2 = ''.join(team2_perm)
                    rounds.append((team1, team2, sitting_player))

    return rounds

def generate_base_cycle():
    """Generate the base 5-round cycle using ranking approach"""
    all_rounds = generate_all_rounds()
    base_cycle = []

    # Generate exactly 5 rounds, one for each sitting player
    for round_num in range(1, 6):
        required_sitting = get_sitting_player(round_num)
        best_score = -1
        best_round = None

        # Only consider candidates where the correct player sits
        valid_candidates = [r for r in all_rounds if r[2] == required_sitting]

        for candidate in valid_candidates:
            score = score_round(candidate, base_cycle, allow_mirrors=False)
            if score > best_score:
                best_score = score
                best_round = candidate

        if best_round is None:
            break

        base_cycle.append(best_round)

    return base_cycle

def generate_schedule(num_rounds):
    """Generate schedule with variety while maintaining sitting rotation"""
    all_rounds = generate_all_rounds()
    schedule = []

    # Generate each round individually with the ranking approach
    for round_num in range(1, num_rounds + 1):
        required_sitting = get_sitting_player(round_num)
        best_score = -1
        best_round = None

        # Only consider candidates where the correct player sits
        valid_candidates = [r for r in all_rounds if r[2] == required_sitting]

        for candidate in valid_candidates:
            # Skip if this exact round already exists
            if candidate in schedule:
                continue

            # Use allow_mirrors=True after we've used up unique encounters
            allow_mirrors = round_num > 60  # Rough estimate of when we need mirrors
            score = score_round(candidate, schedule, allow_mirrors=allow_mirrors)
            if score > best_score:
                best_score = score
                best_round = candidate

        if best_round is None:
            # If no valid round found, we might need to allow mirrors earlier
            for candidate in valid_candidates:
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
        print("Usage: python main5.py <num_rounds>", file=sys.stderr)
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
    for i, (team1, team2, sitting) in enumerate(schedule, 1):
        print(f"{i},{team1},{team2},{sitting}")

if __name__ == "__main__":
    main()