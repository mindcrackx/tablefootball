"""
Table Football Tournament Scheduler using CPMpy
Optimizes for unique encounters and position switching
"""
import cpmpy as cp
import argparse
import sys
from itertools import combinations, permutations
from typing import List, Tuple, Dict, Set


class TableFootballScheduler:
    """
    Scheduler for table football tournaments using constraint programming.

    Notation:
    - Team: "AB" means A=Defense, B=Attack
    - Encounter: ("AB", "CD") means team AB vs team CD
    """

    def __init__(self, players: List[str]):
        """
        Initialize scheduler with list of players.

        Args:
            players: List of player names (supports 4-6 players)
        """
        if len(players) < 4 or len(players) > 6:
            raise ValueError("Supports 4-6 players only")

        self.players = players
        self.num_players = len(players)

        # Pre-calculate all possible teams and encounters
        self.all_teams = self._generate_all_teams()
        self.all_encounters = self._generate_all_encounters()

        # Create mappings for CSP variables
        self.team_to_id = {team: i for i, team in enumerate(self.all_teams)}
        self.encounter_to_id = {enc: i for i, enc in enumerate(self.all_encounters)}

    def _generate_all_teams(self) -> List[str]:
        """Generate all possible teams (2 players each)"""
        teams = []
        for p1, p2 in permutations(self.players, 2):
            teams.append(p1 + p2)  # p1=Defense, p2=Attack
        return sorted(teams)

    def _generate_all_encounters(self) -> List[Tuple[str, str, str]]:
        """Generate all possible encounters (team1 vs team2, sitting_players)"""
        encounters = []

        if self.num_players == 4:
            # For 4 players: no sitting players
            return self._generate_4_player_encounters()
        else:
            # For 5-6 players: some players sit out
            return self._generate_sitting_encounters()

    def _generate_4_player_encounters(self) -> List[Tuple[str, str]]:
        """Generate all possible encounters for 4 players (no sitting)"""
        encounters = []

        # Get all possible 2-player combinations
        player_pairs = list(combinations(self.players, 2))

        # For each way to split 4 players into 2 teams of 2
        for i, team1_players in enumerate(player_pairs):
            # Remaining players form team2
            team2_players = tuple(p for p in self.players if p not in team1_players)

            # Generate all position assignments for both teams
            for t1_p1, t1_p2 in permutations(team1_players, 2):
                for t2_p1, t2_p2 in permutations(team2_players, 2):
                    team1 = t1_p1 + t1_p2  # Defense + Attack
                    team2 = t2_p1 + t2_p2  # Defense + Attack

                    encounter = (team1, team2)
                    encounters.append(encounter)

        return sorted(encounters)

    def _generate_sitting_encounters(self) -> List[Tuple[str, str, str]]:
        """Generate all possible encounters with sitting players (5-6 players)"""
        encounters = []
        sitting_count = self.num_players - 4  # 1 for 5 players, 2 for 6 players

        # Generate all combinations of sitting players
        for sitting_players in combinations(self.players, sitting_count):
            # Remaining 4 players form the game
            playing_players = [p for p in self.players if p not in sitting_players]

            # Generate all possible 2-player team combinations from remaining 4
            player_pairs = list(combinations(playing_players, 2))

            for i, team1_players in enumerate(player_pairs):
                # Remaining players form team2
                team2_players = tuple(p for p in playing_players if p not in team1_players)

                # Generate all position assignments for both teams
                for t1_p1, t1_p2 in permutations(team1_players, 2):
                    for t2_p1, t2_p2 in permutations(team2_players, 2):
                        team1 = t1_p1 + t1_p2  # Defense + Attack
                        team2 = t2_p1 + t2_p2  # Defense + Attack

                        # Create sitting string (sorted for consistency)
                        sitting_str = ''.join(sorted(sitting_players))

                        encounter = (team1, team2, sitting_str)
                        encounters.append(encounter)

        return sorted(encounters)

    def generate_schedule(self, num_rounds: int):
        """
        Generate optimal tournament schedule using CPMpy.

        Args:
            num_rounds: Number of rounds to schedule

        Returns:
            For 4 players: List[Tuple[str, str]] - (team1, team2) for each round
            For 5-6 players: List[Tuple[str, str, str]] - (team1, team2, sitting) for each round
        """
        # Cap rounds at maximum possible unique encounters
        max_possible_rounds = len(self.all_encounters)
        if num_rounds > max_possible_rounds:
            print(f"Warning: Requested {num_rounds} rounds, but only {max_possible_rounds} unique encounters possible.")
            print(f"Generating {max_possible_rounds} rounds instead.")
            num_rounds = max_possible_rounds
        model = cp.Model()

        # Decision variables: which encounter is chosen for each round
        # encounter_vars[r] = index of encounter chosen for round r
        encounter_vars = cp.intvar(0, len(self.all_encounters) - 1, shape=num_rounds, name="encounters")

        # Hard Constraint 1: No exact repetition of encounters
        model += cp.AllDifferent(encounter_vars)

        # Hard Constraint 2: First round standardization
        if self.num_players == 4:
            # 4 players: AB vs CD
            ab_vs_cd_index = self.encounter_to_id[("AB", "CD")]
            if num_rounds == 1:
                model += [encounter_vars == ab_vs_cd_index]
            else:
                model += [encounter_vars[0] == ab_vs_cd_index]
        else:
            # 5-6 players: AB vs CD (E sits, or EF sit)
            sitting_str = ''.join(sorted(self.players[4:]))  # Players beyond first 4
            standard_first = ("AB", "CD", sitting_str)
            if standard_first in self.encounter_to_id:
                first_index = self.encounter_to_id[standard_first]
                if num_rounds == 1:
                    model += [encounter_vars == first_index]
                else:
                    model += [encounter_vars[0] == first_index]

        # Hard Constraint 3: Second round optimization (4 players only for now)
        if num_rounds >= 2 and self.num_players == 4:
            # Only allow the best Round 2 options: DA vs BC or BC vs DA (perfect 5/5 transition quality)
            optimal_round2_encounters = [
                self.encounter_to_id.get(("DA", "BC"), -1),
                self.encounter_to_id.get(("BC", "DA"), -1)
            ]
            # Filter out any that don't exist (-1)
            optimal_round2_encounters = [idx for idx in optimal_round2_encounters if idx != -1]

            if optimal_round2_encounters:
                # Create constraint that round 2 must be one of the truly optimal encounters
                round2_constraint = cp.any([encounter_vars[1] == idx for idx in optimal_round2_encounters])
                model += [round2_constraint]

        # Hard Constraint 4: Sitting fairness (for 5-6 players only)
        if self.num_players > 4:
            self._add_sitting_fairness_constraints(model, encounter_vars, num_rounds)

        # For initial implementation, let's solve without soft constraints
        # and then post-process to find the best solution

        # Solve the model and find the best solution
        solutions = []
        max_solutions = min(500, len(self.all_encounters))  # Increased search space for better optimization

        # Define callback to collect solutions
        def collect_solution():
            # Access array variables correctly
            if num_rounds == 1:
                encounter_indices = [encounter_vars.value()]
            else:
                encounter_indices = [encounter_vars[i].value() for i in range(num_rounds)]
            schedule = [self.all_encounters[idx] for idx in encounter_indices]
            solutions.append(schedule)

        # Find multiple solutions using callback
        num_solutions = model.solveAll(display=collect_solution, solution_limit=max_solutions)

        if num_solutions == 0:
            raise ValueError(f"No valid schedule found for {num_rounds} rounds")

        # Score all solutions and return the best one
        best_schedule = None
        best_score = -1

        for schedule in solutions:
            pos_score = self.calculate_position_switching_score(schedule)
            mix_score = self.calculate_team_mixing_score(schedule)
            side_score = self.calculate_side_balance_score(schedule)
            sitting_score = self.calculate_sitting_fairness_score(schedule)
            # Multi-objective optimization with priorities:
            # 1. Position switching (highest) 2. Team mixing 3. Sitting fairness 4. Side balance (lowest)
            total_score = pos_score * 10 + mix_score * 5 + sitting_score * 3 + side_score * 1

            if total_score > best_score:
                best_score = total_score
                best_schedule = schedule

        return best_schedule

    def _add_sitting_fairness_constraints(self, model, encounter_vars, num_rounds):
        """
        Add sitting fairness constraints to ensure PERFECT rotation of sitting players.
        This is the HIGHEST PRIORITY for 5-6 players - each player must sit equally.
        """
        if self.num_players == 5:
            # 5 players: Perfect rotation - each player sits every 5th round
            # Round 1: E sits, Round 2: A sits, Round 3: B sits, Round 4: C sits, Round 5: D sits, etc.
            sitting_rotation = ['E', 'A', 'B', 'C', 'D']  # Starting rotation after E sits in round 1

            for round_idx in range(num_rounds):
                # Determine who should sit in this round based on perfect rotation
                sitting_player_idx = round_idx % 5
                required_sitting_player = sitting_rotation[sitting_player_idx]

                # Find all encounters where this specific player sits
                valid_encounter_indices = []
                for encounter_idx, encounter in enumerate(self.all_encounters):
                    _, _, sitting_player = encounter
                    if sitting_player == required_sitting_player:
                        valid_encounter_indices.append(encounter_idx)

                if valid_encounter_indices:
                    # This round MUST choose an encounter where the required player sits
                    sitting_constraint = cp.any([encounter_vars[round_idx] == idx for idx in valid_encounter_indices])
                    model += [sitting_constraint]

        elif self.num_players == 6:
            # 6 players: Perfect rotation - each player sits every 3rd round (since 2 sit each round)
            # More complex rotation needed for 6 players with 2 sitting each round

            # For 6 players, we need to ensure each player sits roughly equally
            # Let's use a simpler constraint: each player sits in exactly the right proportion
            sitting_counts = {}
            for player in self.players:
                sitting_counts[player] = cp.intvar(0, num_rounds, name=f"sitting_{player}")

            # For each player, constrain their sitting count
            for player in self.players:
                sitting_sum_terms = []
                for round_idx in range(num_rounds):
                    for encounter_idx, encounter in enumerate(self.all_encounters):
                        _, _, sitting_players = encounter
                        if player in sitting_players:
                            sitting_sum_terms.append(encounter_vars[round_idx] == encounter_idx)

                if sitting_sum_terms:
                    model += [sitting_counts[player] == cp.sum(sitting_sum_terms)]
                else:
                    model += [sitting_counts[player] == 0]

            # Perfect fairness: each player should sit the same number of times (within 1)
            expected_sits = (num_rounds * 2) // 6  # 2 players sit each round, divided by 6 players
            for player in self.players:
                model += [sitting_counts[player] >= expected_sits]
                model += [sitting_counts[player] <= expected_sits + 1]

    def _extract_encounter_info(self, encounter):
        """Extract team1, team2, and sitting info from encounter tuple"""
        if self.num_players == 4:
            team1, team2 = encounter
            sitting = ""
        else:
            team1, team2, sitting = encounter
        return team1, team2, sitting

    def _position_switching_constraint(self, enc1_var, enc2_var):
        """
        Create constraint expression for position switching score between two encounters.
        Returns score 0-4 based on how many players switch positions.
        """
        # This is complex to implement directly in CPMpy due to the need to decode encounter indices
        # For now, return a placeholder - we'll implement this as a post-processing step
        return cp.intvar(0, 4)

    def _team_mixing_constraint(self, enc1_var, enc2_var):
        """
        Create constraint expression for team mixing score between two encounters.
        Returns score 1-5 based on mixing quality.
        """
        # This is complex to implement directly in CPMpy due to the need to decode encounter indices
        # For now, return a placeholder - we'll implement this as a post-processing step
        return cp.intvar(1, 5)

    def calculate_position_switching_score(self, schedule) -> int:
        """
        Calculate position switching score for a complete schedule.

        Args:
            schedule: List of encounters

        Returns:
            Total score for position switching (higher = better)
        """
        if len(schedule) < 2:
            return 0

        total_score = 0

        # Track positions for each player across rounds
        positions = {player: [] for player in self.players}

        for encounter in schedule:
            team1, team2, sitting = self._extract_encounter_info(encounter)

            # Team1: first=Defense, second=Attack
            positions[team1[0]].append('D')
            positions[team1[1]].append('A')
            # Team2: first=Defense, second=Attack
            positions[team2[0]].append('D')
            positions[team2[1]].append('A')

            # Sitting players get 'S' (sitting) marker
            if sitting:
                for player in sitting:
                    positions[player].append('S')

        # Score based on position switches with bonus for perfect rounds
        for i in range(len(schedule) - 1):
            round_switches = 0
            playing_players = 0  # Count how many players are playing (not sitting)

            for player in self.players:
                # Only count switches for players who are playing in both rounds
                if positions[player][i] != 'S' and positions[player][i+1] != 'S':
                    playing_players += 1
                    if positions[player][i] != positions[player][i+1]:  # Position switch
                        round_switches += 1
                        total_score += 1

            # Bonus for perfect switching rounds (all playing players switch)
            if playing_players == 4 and round_switches == 4:
                total_score += 4  # Extra bonus for perfect rounds

        return total_score

    def calculate_sitting_fairness_score(self, schedule) -> int:
        """
        Calculate sitting fairness score - how evenly sitting time is distributed.

        Args:
            schedule: List of encounters

        Returns:
            Sitting fairness score (higher = better fairness)
        """
        if self.num_players == 4:
            return 0  # No sitting in 4-player games

        sitting_count = {player: 0 for player in self.players}

        for encounter in schedule:
            _, _, sitting = self._extract_encounter_info(encounter)
            if sitting:
                for player in sitting:
                    sitting_count[player] += 1

        # Calculate fairness score
        sitting_times = list(sitting_count.values())
        if not sitting_times:
            return 0

        min_sits = min(sitting_times)
        max_sits = max(sitting_times)
        sit_difference = max_sits - min_sits

        # Perfect fairness: everyone sits the same number of times
        # Score decreases as difference increases
        max_possible_score = len(schedule) * self.num_players  # theoretical max
        fairness_score = max(0, max_possible_score - sit_difference * 10)

        return fairness_score

    def calculate_side_balance_score(self, schedule) -> int:
        """
        Calculate side balance score - how evenly teams are distributed across table sides.

        Args:
            schedule: List of encounters

        Returns:
            Side balance score (higher = better balance)
        """
        if len(schedule) == 0:
            return 0

        # Track which side each team plays on
        side_1_count = {}  # Teams that played on side 1 (first position in encounter)
        side_2_count = {}  # Teams that played on side 2 (second position in encounter)

        # Initialize counts for all possible teams
        for team in self.all_teams:
            side_1_count[team] = 0
            side_2_count[team] = 0

        # Count side appearances
        for encounter in schedule:
            team1, team2, sitting = self._extract_encounter_info(encounter)
            side_1_count[team1] += 1
            side_2_count[team2] += 1

        # Calculate balance score
        total_score = 0
        total_teams_playing = 0

        for team in self.all_teams:
            side1_plays = side_1_count[team]
            side2_plays = side_2_count[team]
            total_plays = side1_plays + side2_plays

            if total_plays > 0:
                total_teams_playing += 1
                # Perfect balance: equal plays on both sides
                balance_diff = abs(side1_plays - side2_plays)
                # Score: fewer difference = better (max points when perfectly balanced)
                balance_score = max(0, total_plays - balance_diff)
                total_score += balance_score

        return total_score

    def calculate_team_mixing_score(self, schedule) -> int:
        """
        Calculate team mixing score based on 5-tier hierarchy.

        Args:
            schedule: List of encounters

        Returns:
            Total score for team mixing (higher = better)
        """
        if len(schedule) < 2:
            return 0

        total_score = 0

        for i in range(len(schedule) - 1):
            enc1 = schedule[i]
            enc2 = schedule[i + 1]

            # Calculate transition score
            transition_score = self._calculate_transition_quality(enc1, enc2)
            total_score += transition_score

        return total_score

    def _calculate_transition_quality(self, enc1, enc2) -> int:
        """
        Calculate quality score for transition between two encounters.

        5-tier hierarchy:
        5 = Perfect: all players change teammates and positions
        4 = Very good: all players change teammates
        3 = Good: same teams but positions switched
        2 = Poor: minimal changes
        1 = Forbidden: exact repetition
        """
        if enc1 == enc2:
            return 1  # Forbidden (but shouldn't happen due to constraints)

        # Extract teams and players
        t1_1, t1_2, _ = self._extract_encounter_info(enc1)
        t2_1, t2_2, _ = self._extract_encounter_info(enc2)

        # Get team compositions (ignoring positions)
        teams1 = {frozenset(t1_1), frozenset(t1_2)}
        teams2 = {frozenset(t2_1), frozenset(t2_2)}

        # Check if team compositions are identical
        if teams1 == teams2:
            # Same teams - check if positions switched
            if (t1_1 != t2_1 and t1_1 != t2_2) or (t1_2 != t2_1 and t1_2 != t2_2):
                return 3  # Same teams, positions switched
            else:
                return 2  # Minimal changes
        else:
            # Different team compositions
            # Check if all players changed teammates (only for players who are playing in both rounds)
            players_in_teams1 = self._get_teammates_mapping(enc1)
            players_in_teams2 = self._get_teammates_mapping(enc2)

            all_changed_teammates = True
            playing_both_rounds = []

            # Only consider players who are playing in both rounds
            for player in self.players:
                if player in players_in_teams1 and player in players_in_teams2:
                    playing_both_rounds.append(player)
                    if players_in_teams1[player] == players_in_teams2[player]:
                        all_changed_teammates = False
                        break

            # If no players are playing in both rounds, consider it a good transition
            if not playing_both_rounds:
                all_changed_teammates = True

            if all_changed_teammates:
                # Check if positions also changed (only for players playing both rounds)
                positions1 = self._get_positions(enc1)
                positions2 = self._get_positions(enc2)

                all_changed_positions = all(
                    positions1[player] != positions2[player]
                    for player in playing_both_rounds
                ) if playing_both_rounds else True

                if all_changed_positions:
                    return 5  # Perfect: all change teammates and positions
                else:
                    return 4  # Very good: all change teammates
            else:
                return 2  # Poor: minimal changes

    def _get_teammates_mapping(self, encounter) -> Dict[str, str]:
        """Get mapping of each player to their teammate"""
        team1, team2, _ = self._extract_encounter_info(encounter)
        mapping = {}

        # Team 1
        mapping[team1[0]] = team1[1]  # Defense player's teammate is Attack player
        mapping[team1[1]] = team1[0]  # Attack player's teammate is Defense player

        # Team 2
        mapping[team2[0]] = team2[1]
        mapping[team2[1]] = team2[0]

        return mapping

    def _get_positions(self, encounter) -> Dict[str, str]:
        """Get mapping of each player to their position"""
        team1, team2, _ = self._extract_encounter_info(encounter)
        positions = {}

        # Team 1: first=Defense, second=Attack
        positions[team1[0]] = 'D'
        positions[team1[1]] = 'A'

        # Team 2: first=Defense, second=Attack
        positions[team2[0]] = 'D'
        positions[team2[1]] = 'A'

        return positions

    def print_schedule(self, schedule: List[Tuple[str, str]]) -> None:
        """Print schedule in human-readable format"""
        print("Tournament Schedule:")
        print("=" * 40)
        print("Note: First letter = Defense, Second letter = Attack")
        print()

        for i, encounter in enumerate(schedule, 1):
            team1, team2, sitting = self._extract_encounter_info(encounter)
            if sitting:
                print(f"Round {i}: {team1} vs {team2} ({sitting} sits)")
            else:
                print(f"Round {i}: {team1} vs {team2}")

        # Calculate and display scores
        pos_score = self.calculate_position_switching_score(schedule)
        mix_score = self.calculate_team_mixing_score(schedule)
        side_score = self.calculate_side_balance_score(schedule)
        sitting_score = self.calculate_sitting_fairness_score(schedule)

        print(f"Position Switching Score: {pos_score}")
        print(f"Team Mixing Score: {mix_score}")
        if self.num_players > 4:
            print(f"Sitting Fairness Score: {sitting_score}")
        print(f"Side Balance Score: {side_score}")
        print(f"Total Weighted Score: {pos_score * 10 + mix_score * 5 + sitting_score * 3 + side_score * 1}")

        # Show detailed breakdowns
        if self.num_players > 4:
            self._print_sitting_details(schedule)
        self._print_side_balance_details(schedule)

    def _print_sitting_details(self, schedule) -> None:
        """Print detailed sitting fairness breakdown"""
        if self.num_players == 4:
            return

        print("\nSitting Fairness Analysis:")
        print("-" * 30)

        sitting_count = {player: 0 for player in self.players}

        for encounter in schedule:
            _, _, sitting = self._extract_encounter_info(encounter)
            if sitting:
                for player in sitting:
                    sitting_count[player] += 1

        print("Player | Sits | Fairness")
        print("-" * 30)

        sitting_times = list(sitting_count.values())
        min_sits = min(sitting_times)
        max_sits = max(sitting_times)

        for player in sorted(self.players):
            sits = sitting_count[player]
            difference = sits - min_sits
            fairness_status = "✅" if difference <= 1 else "⚠️"
            print(f"   {player}   |  {sits:2d}  | {fairness_status} (+{difference})")

    def _print_side_balance_details(self, schedule) -> None:
        """Print detailed side balance breakdown"""
        if len(schedule) == 0:
            return

        print("\nSide Balance Analysis:")
        print("-" * 30)

        # Track which side each team plays on
        side_1_count = {}  # Teams that played on side 1 (first position)
        side_2_count = {}  # Teams that played on side 2 (second position)

        # Initialize counts for all possible teams
        for team in self.all_teams:
            side_1_count[team] = 0
            side_2_count[team] = 0

        # Count side appearances
        for encounter in schedule:
            team1, team2, sitting = self._extract_encounter_info(encounter)
            side_1_count[team1] += 1
            side_2_count[team2] += 1

        # Display results
        print("Team | Side 1 | Side 2 | Balance")
        print("-" * 30)

        for team in sorted(self.all_teams):
            side1_plays = side_1_count[team]
            side2_plays = side_2_count[team]
            total_plays = side1_plays + side2_plays

            if total_plays > 0:
                balance_diff = abs(side1_plays - side2_plays)
                balance_status = "✅" if balance_diff <= 1 else "⚠️"
                print(f" {team}   |   {side1_plays:2d}   |   {side2_plays:2d}   | {balance_status} ({balance_diff})")


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Generate optimal table football tournament schedules')
    parser.add_argument('--players', '-p', type=int, default=5, choices=[4, 5, 6],
                        help='Number of players (default: 5)')
    parser.add_argument('--rounds', '-r', type=int, default=50,
                        help='Number of rounds to generate (default: 50)')

    args = parser.parse_args()

    # Create player list
    player_names = [chr(ord('A') + i) for i in range(args.players)]
    scheduler = TableFootballScheduler(player_names)

    print(f"🏓 Table Football Tournament Scheduler")
    print(f"Players: {args.players} ({', '.join(player_names)})")
    print(f"Maximum possible unique encounters: {len(scheduler.all_encounters)}")
    print()

    # Generate schedule
    try:
        schedule = scheduler.generate_schedule(args.rounds)
        scheduler.print_schedule(schedule)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()