#!/usr/bin/env python3
"""
Improved Table Football Tournament Scheduler
Focuses on optimal team mixing and position switching with CSV output
"""

import csv
import argparse
from typing import List, Tuple, Dict, Set
from itertools import combinations, permutations


class ImprovedTableFootballScheduler:
    """
    Improved scheduler focusing on optimal patterns:
    - Perfect sitting rotation (each player sits 1/5 rounds)
    - Complete team mixing (avoid same teammates)
    - Position switching (Defense <-> Attack)
    """

    def __init__(self, players: List[str]):
        if len(players) < 4 or len(players) > 6:
            raise ValueError("Supports 4-6 players only")

        self.players = players
        self.num_players = len(players)

    def generate_schedule(self, num_rounds: int) -> List[Tuple[int, str, str, str, str, str]]:
        """
        Generate optimal schedule using greedy approach with backtracking.

        Returns:
            List of tuples: (round, team1_defense, team1_attack, team2_defense, team2_attack, sitting_players)
        """
        schedule = []

        if self.num_players == 4:
            return self._generate_4_player_schedule(num_rounds)
        else:
            return self._generate_5_6_player_schedule(num_rounds)

    def _generate_4_player_schedule(self, num_rounds: int) -> List[Tuple[int, str, str, str, str, str]]:
        """Generate schedule for 4 players (no sitting)"""
        schedule = []

        # Track player positions across rounds
        player_positions = {player: [] for player in self.players}

        for round_num in range(1, num_rounds + 1):
            if round_num == 1:
                # Standard first round: AB vs CD
                encounter = (round_num, 'A', 'B', 'C', 'D', '')
            else:
                # Find best next encounter
                encounter = self._find_best_4_player_encounter(round_num, schedule, player_positions)

            schedule.append(encounter)

            # Update position tracking
            _, t1_def, t1_att, t2_def, t2_att, _ = encounter
            player_positions[t1_def].append('D')
            player_positions[t1_att].append('A')
            player_positions[t2_def].append('D')
            player_positions[t2_att].append('A')

        return schedule

    def _generate_5_6_player_schedule(self, num_rounds: int) -> List[Tuple[int, str, str, str, str, str]]:
        """Generate schedule for 5-6 players with sitting rotation"""
        schedule = []

        # Track player states
        player_positions = {player: [] for player in self.players}  # 'D', 'A', 'S' for Defense, Attack, Sitting
        player_teammates = {player: [] for player in self.players}  # Track who they've played with
        sitting_counts = {player: 0 for player in self.players}

        for round_num in range(1, num_rounds + 1):
            if round_num == 1:
                # Standard first round: AB vs CD (E sits for 5 players, EF sit for 6 players)
                sitting_players = ''.join(self.players[4:])  # E for 5 players, EF for 6 players
                encounter = (round_num, 'A', 'B', 'C', 'D', sitting_players)
            else:
                # Find optimal next encounter
                encounter = self._find_best_encounter(round_num, schedule, player_positions,
                                                   player_teammates, sitting_counts)

            schedule.append(encounter)

            # Update tracking
            self._update_player_tracking(encounter, player_positions, player_teammates, sitting_counts)

        return schedule

    def _find_best_4_player_encounter(self, round_num: int, schedule: List,
                                    player_positions: Dict) -> Tuple[int, str, str, str, str, str]:
        """Find best encounter for 4 players focusing on position switching and team mixing"""

        # Generate all possible encounters
        possible_encounters = []
        for team1_players in combinations(self.players, 2):
            team2_players = [p for p in self.players if p not in team1_players]

            # All position combinations
            for t1_def, t1_att in permutations(team1_players, 2):
                for t2_def, t2_att in permutations(team2_players, 2):
                    encounter = (round_num, t1_def, t1_att, t2_def, t2_att, '')
                    possible_encounters.append(encounter)

        # Score each encounter
        best_encounter = None
        best_score = -1

        for encounter in possible_encounters:
            score = self._score_4_player_encounter(encounter, schedule, player_positions)
            if score > best_score:
                best_score = score
                best_encounter = encounter

        return best_encounter

    def _find_best_encounter(self, round_num: int, schedule: List, player_positions: Dict,
                           player_teammates: Dict, sitting_counts: Dict) -> Tuple[int, str, str, str, str, str]:
        """Find best encounter for 5-6 players with comprehensive scoring"""

        possible_encounters = []
        sitting_player_count = self.num_players - 4

        # Generate all possible sitting combinations
        for sitting_players_tuple in combinations(self.players, sitting_player_count):
            sitting_str = ''.join(sorted(sitting_players_tuple))
            playing_players = [p for p in self.players if p not in sitting_players_tuple]

            # Generate team combinations from playing players
            for team1_players in combinations(playing_players, 2):
                team2_players = [p for p in playing_players if p not in team1_players]

                # All position combinations
                for t1_def, t1_att in permutations(team1_players, 2):
                    for t2_def, t2_att in permutations(team2_players, 2):
                        encounter = (round_num, t1_def, t1_att, t2_def, t2_att, sitting_str)
                        possible_encounters.append(encounter)

        # Score and select best encounter
        best_encounter = None
        best_score = -1

        for encounter in possible_encounters:
            score = self._score_encounter(encounter, schedule, player_positions,
                                        player_teammates, sitting_counts)
            if score > best_score:
                best_score = score
                best_encounter = encounter

        return best_encounter

    def _score_encounter(self, encounter: Tuple, schedule: List, player_positions: Dict,
                        player_teammates: Dict, sitting_counts: Dict) -> float:
        """Comprehensive scoring for encounter quality"""
        if not schedule:  # First round
            return 1000  # Always good

        round_num, t1_def, t1_att, t2_def, t2_att, sitting = encounter

        score = 0

        # 1. Sitting fairness (CRITICAL for 5-6 players)
        sitting_score = self._calculate_sitting_fairness_score(encounter, sitting_counts)
        score += sitting_score * 100  # Very high priority

        # 2. Position switching score
        position_score = self._calculate_position_switching_score(encounter, player_positions)
        score += position_score * 50

        # 3. Team mixing score (avoid same teammates)
        team_mixing_score = self._calculate_team_mixing_score(encounter, player_teammates)
        score += team_mixing_score * 75

        # 4. Perfect rotation bonus (every 5th round for 5 players)
        if self.num_players == 5:
            expected_sitting_player = self.players[(round_num - 1) % 5]
            if sitting == expected_sitting_player:
                score += 200  # Big bonus for perfect rotation

        return score

    def _score_4_player_encounter(self, encounter: Tuple, schedule: List,
                                player_positions: Dict) -> float:
        """Scoring for 4-player encounters"""
        if not schedule:
            return 1000

        score = 0

        # Position switching
        position_score = self._calculate_position_switching_score(encounter, player_positions)
        score += position_score * 50

        # Team mixing (different from previous round)
        if schedule:
            prev_encounter = schedule[-1]
            prev_teams = {frozenset([prev_encounter[1], prev_encounter[2]]),
                         frozenset([prev_encounter[3], prev_encounter[4]])}
            curr_teams = {frozenset([encounter[1], encounter[2]]),
                         frozenset([encounter[3], encounter[4]])}

            if prev_teams != curr_teams:
                score += 100  # Bonus for different team compositions

        return score

    def _calculate_sitting_fairness_score(self, encounter: Tuple, sitting_counts: Dict) -> float:
        """Calculate sitting fairness score"""
        _, _, _, _, _, sitting = encounter
        if not sitting:
            return 50  # Neutral for 4-player games

        # Check if sitting players have sat the least
        current_min_sits = min(sitting_counts.values())

        score = 0
        for player in sitting:
            if sitting_counts[player] == current_min_sits:
                score += 25  # Bonus for giving turn to player who has sat least
            else:
                score -= 10  # Penalty for making someone sit again too soon

        return score

    def _calculate_position_switching_score(self, encounter: Tuple, player_positions: Dict) -> float:
        """Calculate position switching score"""
        _, t1_def, t1_att, t2_def, t2_att, sitting = encounter

        score = 0
        playing_players = [t1_def, t1_att, t2_def, t2_att]

        for player in playing_players:
            if player_positions[player]:  # Has played before
                last_position = player_positions[player][-1]
                if last_position == 'S':  # Was sitting, any position is good
                    score += 10
                elif last_position == 'D' and player in [t1_att, t2_att]:  # Defense -> Attack
                    score += 20
                elif last_position == 'A' and player in [t1_def, t2_def]:  # Attack -> Defense
                    score += 20
                # No penalty for same position, just no bonus

        return score

    def _calculate_team_mixing_score(self, encounter: Tuple, player_teammates: Dict) -> float:
        """Calculate team mixing score - heavily penalize same teammates"""
        _, t1_def, t1_att, t2_def, t2_att, _ = encounter

        team1 = frozenset([t1_def, t1_att])
        team2 = frozenset([t2_def, t2_att])

        score = 100  # Start with full score

        # Check if these team compositions have been used before
        for player in [t1_def, t1_att]:
            teammate = t1_att if player == t1_def else t1_def
            if teammate in player_teammates[player]:
                score -= 30  # Heavy penalty for repeated teammates

        for player in [t2_def, t2_att]:
            teammate = t2_att if player == t2_def else t2_def
            if teammate in player_teammates[player]:
                score -= 30  # Heavy penalty for repeated teammates

        return max(0, score)

    def _update_player_tracking(self, encounter: Tuple, player_positions: Dict,
                              player_teammates: Dict, sitting_counts: Dict):
        """Update all player tracking after an encounter"""
        _, t1_def, t1_att, t2_def, t2_att, sitting = encounter

        # Update positions
        player_positions[t1_def].append('D')
        player_positions[t1_att].append('A')
        player_positions[t2_def].append('D')
        player_positions[t2_att].append('A')

        # Update teammates
        player_teammates[t1_def].append(t1_att)
        player_teammates[t1_att].append(t1_def)
        player_teammates[t2_def].append(t2_att)
        player_teammates[t2_att].append(t2_def)

        # Update sitting
        for player in sitting:
            player_positions[player].append('S')
            sitting_counts[player] += 1

    def export_to_csv(self, schedule: List[Tuple], filename: str):
        """Export schedule to CSV format"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['round', 'team1', 'team2', 'sitting'])

            # Write schedule
            for encounter in schedule:
                round_num, t1_def, t1_att, t2_def, t2_att, sitting = encounter
                team1 = t1_def + t1_att
                team2 = t2_def + t2_att
                writer.writerow([round_num, team1, team2, sitting])

    def print_csv_to_stdout(self, schedule: List[Tuple]):
        """Print schedule in CSV format to stdout"""
        print("round,team1,team2,sitting")

        for encounter in schedule:
            round_num, t1_def, t1_att, t2_def, t2_att, sitting = encounter
            team1 = t1_def + t1_att
            team2 = t2_def + t2_att
            print(f"{round_num},{team1},{team2},{sitting}")

    def print_schedule_summary(self, schedule: List[Tuple], verbose: bool = False):
        """Print a summary of the schedule"""
        print(f"Players: {self.num_players} ({', '.join(self.players)})")
        print(f"Rounds: {len(schedule)}")
        print()

        print(f"Schedule Summary ({len(schedule)} rounds):")
        print("=" * 50)

        if verbose:
            # Show all rounds
            for encounter in schedule:
                round_num, t1_def, t1_att, t2_def, t2_att, sitting = encounter
                if sitting:
                    print(f"Round {round_num}: {t1_def}{t1_att} vs {t2_def}{t2_att} ({sitting} sits)")
                else:
                    print(f"Round {round_num}: {t1_def}{t1_att} vs {t2_def}{t2_att}")
        else:
            # Show first 3 rounds only
            for encounter in schedule[:3]:
                round_num, t1_def, t1_att, t2_def, t2_att, sitting = encounter
                if sitting:
                    print(f"Round {round_num}: {t1_def}{t1_att} vs {t2_def}{t2_att} ({sitting} sits)")
                else:
                    print(f"Round {round_num}: {t1_def}{t1_att} vs {t2_def}{t2_att}")

            if len(schedule) > 3:
                print(f"... and {len(schedule) - 3} more rounds")

        # Calculate statistics
        self._print_statistics(schedule)

        # Add team distribution for verbose mode
        if verbose:
            self._print_team_distribution(schedule)

    def _print_statistics(self, schedule: List[Tuple]):
        """Print schedule statistics"""
        print("\nStatistics:")
        print("-" * 30)

        # Sitting fairness (for 5-6 players)
        if self.num_players > 4:
            sitting_counts = {player: 0 for player in self.players}
            for encounter in schedule:
                sitting = encounter[5]
                for player in sitting:
                    sitting_counts[player] += 1

            print("Sitting distribution:")
            for player in sorted(self.players):
                count = sitting_counts[player]
                percentage = (count / len(schedule)) * 100
                print(f"  {player}: {count} times ({percentage:.1f}%)")

        # Position switching analysis
        position_switches = 0
        total_transitions = 0

        player_positions = {player: [] for player in self.players}

        for encounter in schedule:
            _, t1_def, t1_att, t2_def, t2_att, sitting = encounter

            # Track positions
            player_positions[t1_def].append('D')
            player_positions[t1_att].append('A')
            player_positions[t2_def].append('D')
            player_positions[t2_att].append('A')

            for player in sitting:
                player_positions[player].append('S')

        # Count position switches
        for player, positions in player_positions.items():
            for i in range(1, len(positions)):
                if positions[i-1] != 'S' and positions[i] != 'S':  # Both playing rounds
                    total_transitions += 1
                    if positions[i-1] != positions[i]:  # Position switched
                        position_switches += 1

        if total_transitions > 0:
            switch_rate = (position_switches / total_transitions) * 100
            print(f"Position switching rate: {position_switches}/{total_transitions} ({switch_rate:.1f}%)")

    def _print_team_distribution(self, schedule: List[Tuple]):
        """Print team distribution analysis"""
        team_counts = {}

        for encounter in schedule:
            _, t1_def, t1_att, t2_def, t2_att, _ = encounter
            team1 = t1_def + t1_att
            team2 = t2_def + t2_att

            team_counts[team1] = team_counts.get(team1, 0) + 1
            team_counts[team2] = team_counts.get(team2, 0) + 1

        print("\nTeam distribution:")
        for team in sorted(team_counts.keys()):
            count = team_counts[team]
            percentage = (count / len(schedule)) * 100
            print(f"  {team}: {count} times ({percentage:.1f}%)")


def main():
    """Main function with command line support"""
    parser = argparse.ArgumentParser(description='Improved Table Football Tournament Scheduler')
    parser.add_argument('--players', '-p', type=int, default=5, choices=[4, 5, 6],
                        help='Number of players (default: 5)')
    parser.add_argument('--rounds', '-r', type=int, default=20,
                        help='Number of rounds (default: 20)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose output with full schedule and statistics')

    args = parser.parse_args()

    # Create player list
    player_names = [chr(ord('A') + i) for i in range(args.players)]
    scheduler = ImprovedTableFootballScheduler(player_names)

    # Generate schedule
    schedule = scheduler.generate_schedule(args.rounds)

    # Always export to tournament.csv
    scheduler.export_to_csv(schedule, 'tournament.csv')

    if args.verbose:
        # Show detailed summary
        scheduler.print_schedule_summary(schedule, verbose=True)
    else:
        # Output CSV to stdout by default
        scheduler.print_csv_to_stdout(schedule)


if __name__ == "__main__":
    main()