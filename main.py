"""
Table Football Tournament Scheduler using CPMpy
Optimizes for unique encounters and position switching
"""
import cpmpy as cp
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
            players: List of player names (currently supports 4 players)
        """
        if len(players) != 4:
            raise ValueError("Currently only supports 4 players")

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

    def _generate_all_encounters(self) -> List[Tuple[str, str]]:
        """Generate all possible encounters (team1 vs team2)"""
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

    def generate_schedule(self, num_rounds: int) -> List[Tuple[str, str]]:
        """
        Generate optimal tournament schedule using CPMpy.

        Args:
            num_rounds: Number of rounds to schedule

        Returns:
            List of encounters (team1, team2) for each round
        """
        model = cp.Model()

        # Decision variables: which encounter is chosen for each round
        # encounter_vars[r] = index of encounter chosen for round r
        encounter_vars = cp.intvar(0, len(self.all_encounters) - 1, shape=num_rounds, name="encounters")

        # Hard Constraint 1: No exact repetition of encounters
        model += cp.AllDifferent(encounter_vars)

        # Hard Constraint 2: First round must always be AB vs CD
        ab_vs_cd_index = self.encounter_to_id[("AB", "CD")]
        if num_rounds == 1:
            model += [encounter_vars == ab_vs_cd_index]
        else:
            model += [encounter_vars[0] == ab_vs_cd_index]

        # Hard Constraint 3: Second round should prioritize perfect position switching
        if num_rounds >= 2:
            # Prefer encounters that give all 4 players position switches
            optimal_round2_encounters = [
                self.encounter_to_id.get(("DA", "BC"), -1),
                self.encounter_to_id.get(("BC", "DA"), -1),
                self.encounter_to_id.get(("BA", "DC"), -1),
                self.encounter_to_id.get(("DC", "BA"), -1)
            ]
            # Filter out any that don't exist (-1)
            optimal_round2_encounters = [idx for idx in optimal_round2_encounters if idx != -1]

            if optimal_round2_encounters:
                # Create constraint that round 2 must be one of the optimal encounters
                round2_constraint = cp.any([encounter_vars[1] == idx for idx in optimal_round2_encounters])
                model += [round2_constraint]

        # Hard Constraint 3: All encounters must be valid (automatically satisfied by domain)

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
            # Increased weight for position switching to prioritize perfect switches
            total_score = mix_score * 3 + pos_score * 5

            if total_score > best_score:
                best_score = total_score
                best_schedule = schedule

        return best_schedule

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

    def calculate_position_switching_score(self, schedule: List[Tuple[str, str]]) -> int:
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

        for team1, team2 in schedule:
            # Team1: first=Defense, second=Attack
            positions[team1[0]].append('D')
            positions[team1[1]].append('A')
            # Team2: first=Defense, second=Attack
            positions[team2[0]].append('D')
            positions[team2[1]].append('A')

        # Score based on position switches with bonus for perfect rounds
        for i in range(len(schedule) - 1):
            round_switches = 0
            for player in self.players:
                if positions[player][i] != positions[player][i+1]:  # Position switch
                    round_switches += 1
                    total_score += 1

            # Bonus for perfect switching rounds (all 4 players switch)
            if round_switches == 4:
                total_score += 4  # Extra bonus for perfect rounds

        return total_score

    def calculate_team_mixing_score(self, schedule: List[Tuple[str, str]]) -> int:
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

    def _calculate_transition_quality(self, enc1: Tuple[str, str], enc2: Tuple[str, str]) -> int:
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
        t1_1, t1_2 = enc1
        t2_1, t2_2 = enc2

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
            # Check if all players changed teammates
            players_in_teams1 = self._get_teammates_mapping(enc1)
            players_in_teams2 = self._get_teammates_mapping(enc2)

            all_changed_teammates = True
            for player in self.players:
                if players_in_teams1[player] == players_in_teams2[player]:
                    all_changed_teammates = False
                    break

            if all_changed_teammates:
                # Check if positions also changed
                positions1 = self._get_positions(enc1)
                positions2 = self._get_positions(enc2)

                all_changed_positions = all(
                    positions1[player] != positions2[player]
                    for player in self.players
                )

                if all_changed_positions:
                    return 5  # Perfect: all change teammates and positions
                else:
                    return 4  # Very good: all change teammates
            else:
                return 2  # Poor: minimal changes

    def _get_teammates_mapping(self, encounter: Tuple[str, str]) -> Dict[str, str]:
        """Get mapping of each player to their teammate"""
        team1, team2 = encounter
        mapping = {}

        # Team 1
        mapping[team1[0]] = team1[1]  # Defense player's teammate is Attack player
        mapping[team1[1]] = team1[0]  # Attack player's teammate is Defense player

        # Team 2
        mapping[team2[0]] = team2[1]
        mapping[team2[1]] = team2[0]

        return mapping

    def _get_positions(self, encounter: Tuple[str, str]) -> Dict[str, str]:
        """Get mapping of each player to their position"""
        team1, team2 = encounter
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

        for i, (team1, team2) in enumerate(schedule, 1):
            print(f"Round {i}: {team1} vs {team2}")

            # Show positions
            positions = self._get_positions((team1, team2))
            pos_str = ", ".join(f"{p}={positions[p]}" for p in sorted(positions.keys()))
            print(f"         Positions: {pos_str}")
            print()

        # Calculate and display scores
        pos_score = self.calculate_position_switching_score(schedule)
        mix_score = self.calculate_team_mixing_score(schedule)

        print(f"Position Switching Score: {pos_score}")
        print(f"Team Mixing Score: {mix_score}")
        print(f"Total Weighted Score: {mix_score * 3 + pos_score * 5}")


def main():
    """Example usage"""
    players = ['A', 'B', 'C', 'D']
    scheduler = TableFootballScheduler(players)

    # Generate schedule for 24 rounds (maximum possible unique encounters)
    try:
        schedule = scheduler.generate_schedule(24)
        scheduler.print_schedule(schedule)
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()