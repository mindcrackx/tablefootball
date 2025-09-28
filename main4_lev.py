#!/usr/bin/env python3

import sys
import itertools

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def hamming_distance(s1, s2):
    """Calculate Hamming distance between two equal-length strings"""
    if len(s1) != len(s2):
        return float('inf')  # Invalid for different lengths
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def normalize_encounter(team1, team2):
    """Normalize encounters so AB vs CD = CD vs AB (same matchup, different sides)"""
    return (team1, team2) if team1 <= team2 else (team2, team1)

def encode_round_positional(team1, team2):
    """Encode round as positional string: ABCD where A=pos1, B=pos2, C=pos3, D=pos4"""
    return team1 + team2

def encode_round_role_explicit(team1, team2):
    """Encode round with explicit roles: A1B2C1D2 where 1=defense, 2=attack"""
    return f"{team1[0]}1{team1[1]}2{team2[0]}1{team2[1]}2"

def encode_round_position_priority(team1, team2):
    """Encode prioritizing position changes: DefenseAttackDefenseAttack"""
    # This encoding makes position swaps have maximum distance
    return f"D{team1[0]}A{team1[1]}D{team2[0]}A{team2[1]}"

def get_player_position(player, team1, team2):
    """Returns (team_index, position) where team_index: 0=team1, 1=team2; position: 0=Defense, 1=Attack"""
    if player in team1:
        return (0, team1.index(player))
    else:
        return (1, team2.index(player))

def calculate_position_change_distance(candidate, previous_rounds):
    """Calculate distance based on actual position changes - domain-specific metric"""
    if not previous_rounds:
        return 1000

    team1, team2 = candidate
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

    # Convert to distance: perfect transition (4 pos changes + 2 team changes) = max distance
    # This mirrors the main4.py scoring logic
    if position_changes == 4:
        base_distance = 100
        if team_changes == 2:
            base_distance += 10  # Bonus for team composition change
    else:
        base_distance = position_changes * 20 + team_changes * 10

    return base_distance

def encode_round_normalized(team1, team2):
    """Encode round in normalized form (smaller team first)"""
    normalized = normalize_encounter(team1, team2)
    return normalized[0] + normalized[1]

def encode_round_team_structure(team1, team2):
    """Encode round preserving team structure: AB-CD"""
    return f"{team1}-{team2}"

def calculate_distance_score(candidate, previous_rounds, encoding_func, distance_func, weight_func):
    """Calculate total weighted distance score for a candidate round"""
    if not previous_rounds:
        return 1000  # High score for first round

    team1, team2 = candidate
    candidate_str = encoding_func(team1, team2)

    total_score = 0
    for i, (prev_team1, prev_team2) in enumerate(previous_rounds):
        prev_str = encoding_func(prev_team1, prev_team2)
        distance = distance_func(candidate_str, prev_str)
        weight = weight_func(i, len(previous_rounds))
        total_score += distance * weight

    return total_score

def weight_recent_heavy(index, total_rounds):
    """Weight function that heavily weights recent rounds"""
    recency = total_rounds - index
    return recency * recency  # Quadratic weighting for recent rounds

def weight_exponential_decay(index, total_rounds):
    """Exponential decay weighting - recent rounds much more important"""
    recency = total_rounds - index
    return 2 ** recency

def weight_linear(index, total_rounds):
    """Simple linear weighting"""
    return total_rounds - index

def score_round_comprehensive(candidate, previous_rounds):
    """Comprehensive scoring using multiple distance metrics and encodings"""
    if not previous_rounds:
        return 1000

    # Strategy 1: Domain-specific position change distance (highest priority)
    position_distance = calculate_position_change_distance(candidate, previous_rounds)

    # Strategy 2: Positional encoding with Levenshtein distance for overall variety
    variety_score = calculate_distance_score(
        candidate, previous_rounds,
        encode_round_positional, levenshtein_distance, weight_recent_heavy
    )

    # Strategy 3: Role-explicit encoding for position-aware variety
    role_score = calculate_distance_score(
        candidate, previous_rounds,
        encode_round_position_priority, levenshtein_distance, weight_exponential_decay
    )

    # Strategy 4: Normalized encoding to discourage encounter repetition
    encounter_score = calculate_distance_score(
        candidate, previous_rounds,
        encode_round_normalized, hamming_distance, weight_linear
    )

    # Strategy 5: Hamming distance on positional for exact position differences
    hamming_score = calculate_distance_score(
        candidate, previous_rounds,
        encode_round_positional, hamming_distance, weight_recent_heavy
    )

    # Combine scores with position changes as the dominant factor (like main4.py)
    combined_score = (
        position_distance * 10.0 +  # Position changes (dominant factor)
        variety_score * 2.0 +       # General variety
        role_score * 1.5 +          # Position-aware variety
        hamming_score * 1.5 +       # Exact position differences
        encounter_score * 1.0       # Encounter uniqueness
    )

    return combined_score

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

def has_duplicate_encounter(candidate, previous_rounds, max_check_rounds=12):
    """Check if candidate creates a duplicate normalized encounter in recent rounds"""
    if len(previous_rounds) >= max_check_rounds:
        return False  # Allow mirrors after max_check_rounds (like main4.py after round 12)

    candidate_normalized = normalize_encounter(candidate[0], candidate[1])

    for prev_round in previous_rounds:
        prev_normalized = normalize_encounter(prev_round[0], prev_round[1])
        if candidate_normalized == prev_normalized:
            return True

    return False

def generate_schedule(num_rounds):
    """Generate optimal schedule using distance-based ranking approach with encounter uniqueness"""
    all_rounds = generate_all_rounds()
    schedule = []

    for round_num in range(1, num_rounds + 1):
        best_score = -1
        best_round = None

        # Filter candidates to avoid encounter duplicates in first 12 rounds
        valid_candidates = []
        for candidate in all_rounds:
            if candidate in schedule:
                continue  # Skip already used rounds

            # Apply encounter uniqueness constraint for first 12 rounds (like main4.py)
            if has_duplicate_encounter(candidate, schedule, max_check_rounds=12):
                continue  # Skip duplicate encounters

            valid_candidates.append(candidate)

        # Score valid candidates
        for candidate in valid_candidates:
            score = score_round_comprehensive(candidate, schedule)
            if score > best_score:
                best_score = score
                best_round = candidate

        if best_round is None:
            break

        schedule.append(best_round)

    return schedule

def main():
    if len(sys.argv) != 2:
        print("Usage: python main4_lev.py <num_rounds>", file=sys.stderr)
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