#!/usr/bin/env python3

def analyze_transitions(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    def get_player_position(player, team1, team2):
        if player in team1:
            return (0, team1.index(player))
        else:
            return (1, team2.index(player))

    perfect_transitions = 0
    total_transitions = 0

    print('Transition analysis:')
    for i in range(1, min(len(lines), 12)):  # First 12 rounds
        curr_team1, curr_team2 = lines[i].strip().split(',')[1:3]
        prev_team1, prev_team2 = lines[i-1].strip().split(',')[1:3]

        position_changes = 0
        team_changes = 0

        for player in ['A', 'B', 'C', 'D']:
            prev_team_idx, prev_pos = get_player_position(player, prev_team1, prev_team2)
            curr_team_idx, curr_pos = get_player_position(player, curr_team1, curr_team2)

            if curr_pos != prev_pos:
                position_changes += 1
            if curr_team_idx != prev_team_idx:
                team_changes += 1

        status = '✓ PERFECT' if position_changes == 4 else f'  {position_changes}/4'

        if position_changes == 4:
            perfect_transitions += 1

        total_transitions += 1

        print(f'Round {i:2d} -> {i+1:2d}: {prev_team1},{prev_team2} -> {curr_team1},{curr_team2} | {position_changes} pos, {team_changes} team | {status}')

    return perfect_transitions, total_transitions

print('=== ORIGINAL main4.py (tournaments/4p.csv) ===')
perfect_orig, total_orig = analyze_transitions('tournaments/4p.csv')
print(f'Perfect transitions: {perfect_orig}/{total_orig} ({perfect_orig/total_orig*100:.1f}%)')

print()
print('=== DISTANCE-BASED main4_lev.py (/tmp/4p.csv) ===')
perfect_lev, total_lev = analyze_transitions('/tmp/4p.csv')
print(f'Perfect transitions: {perfect_lev}/{total_lev} ({perfect_lev/total_lev*100:.1f}%)')

print()
print('=== FINAL VERDICT ===')
if perfect_lev > perfect_orig:
    print('🎉 Distance-based algorithm is BETTER!')
    print(f'   +{perfect_lev - perfect_orig} more perfect transitions')
elif perfect_lev == perfect_orig:
    print('≈ Both algorithms have EQUAL quality')
    print('   Same number of perfect transitions')
else:
    print('❌ Original algorithm is better')
    print(f'   -{perfect_orig - perfect_lev} fewer perfect transitions')