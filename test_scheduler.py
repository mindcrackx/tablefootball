"""
Test cases for table football tournament scheduling (4 players)
"""
import pytest
from main import TableFootballScheduler


class TestTableFootballScheduler:
    """Test cases for 4-player table football scheduling"""

    def setup_method(self):
        """Setup test fixtures"""
        self.players = ['A', 'B', 'C', 'D']
        self.scheduler = TableFootballScheduler(self.players)

    def test_valid_team_notation(self):
        """Test that team notation follows Defense-Attack pattern"""
        team = "AB"
        assert len(team) == 2
        # First letter = Defense, Second letter = Attack
        defense, attack = team[0], team[1]
        assert defense in self.players
        assert attack in self.players
        assert defense != attack

    def test_encounter_creation(self):
        """Test creation of encounters (team1 vs team2)"""
        encounter = ("AB", "CD")
        team1, team2 = encounter

        # Teams should have exactly 2 players each
        assert len(team1) == 2
        assert len(team2) == 2

        # All 4 players should be different
        all_players = set(team1 + team2)
        assert len(all_players) == 4
        assert all_players == set(self.players)

    def test_no_exact_repetition(self):
        """Test that exact encounters cannot repeat"""
        schedule = [
            ("AB", "CD"),
            ("CA", "DB"),
            ("AD", "BC")
        ]

        # No exact repetition should occur
        encounters = set(schedule)
        assert len(encounters) == len(schedule)

        # Also check reverse encounters (AB vs CD == CD vs AB)
        normalized = []
        for t1, t2 in schedule:
            # Normalize by sorting teams
            encounter = tuple(sorted([t1, t2]))
            normalized.append(encounter)

        assert len(set(normalized)) == len(normalized)

    def test_position_switching_validation(self):
        """Test that position switching works correctly"""
        schedule = [
            ("AB", "CD"),  # A=Def, B=Att, C=Def, D=Att
            ("CA", "DB"),  # C=Def, A=Att, D=Def, B=Att
            ("AD", "BC")   # A=Def, D=Att, B=Def, C=Att
        ]

        # Track positions for each player across rounds
        positions = {player: [] for player in self.players}

        for team1, team2 in schedule:
            # Team1: first=Defense, second=Attack
            positions[team1[0]].append('D')  # Defense
            positions[team1[1]].append('A')  # Attack
            # Team2: first=Defense, second=Attack
            positions[team2[0]].append('D')  # Defense
            positions[team2[1]].append('A')  # Attack

        # Check that most players switch positions when possible
        # Note: Perfect alternation may not always be possible due to team constraints
        switches = 0
        total_transitions = 0

        for player, pos_list in positions.items():
            for i in range(len(pos_list) - 1):
                total_transitions += 1
                if pos_list[i] != pos_list[i+1]:  # Position switch
                    switches += 1

        # At least 50% of transitions should be position switches in a good schedule
        if total_transitions > 0:
            switch_rate = switches / total_transitions
            assert switch_rate >= 0.5, f"Position switch rate too low: {switch_rate:.2f} (expected >= 0.5)"

    def test_team_mixing_quality(self):
        """Test team mixing quality according to 5-tier hierarchy"""
        # Perfect mixing: all players change teammates
        round1 = ("AB", "CD")
        round2 = ("CA", "DB")  # A pairs with C instead of B, etc.

        r1_teams = set([frozenset("AB"), frozenset("CD")])
        r2_teams = set([frozenset("CA"), frozenset("DB")])

        # No team should be identical
        assert r1_teams.isdisjoint(r2_teams)

    def test_known_good_schedule(self):
        """Test against known good schedule from PROBLEM.md"""
        expected_schedule = [
            ("AB", "CD"),  # Round 1
            ("CA", "DB"),  # Round 2
            ("AD", "BC")   # Round 3
        ]

        # Validate this schedule meets all constraints
        for i, (team1, team2) in enumerate(expected_schedule):
            # All players covered
            all_players = set(team1 + team2)
            assert all_players == set(self.players), f"Round {i+1} missing players"

            # Valid team sizes
            assert len(team1) == 2 and len(team2) == 2

            # No player plays twice in same round
            assert len(set(team1)) == 2 and len(set(team2)) == 2

    def test_first_round_always_ab_vs_cd(self):
        """Test that first round is always AB vs CD"""
        for rounds in [1, 3, 5, 8]:
            schedule = self.scheduler.generate_schedule(rounds)
            first_round = schedule[0]
            assert first_round == ("AB", "CD"), f"First round should be AB vs CD, got {first_round}"

    def test_minimum_rounds(self):
        """Test edge case: 1-2 rounds"""
        # Should be able to generate valid 1-round schedule
        schedule_1_round = self.scheduler.generate_schedule(1)
        assert len(schedule_1_round) == 1

        # Should be able to generate valid 2-round schedule
        schedule_2_rounds = self.scheduler.generate_schedule(2)
        assert len(schedule_2_rounds) == 2

        # No exact repetitions
        assert schedule_2_rounds[0] != schedule_2_rounds[1]

    def test_maximum_theoretical_encounters(self):
        """Test maximum unique encounters for 4 players"""
        # For 4 players, there are limited unique encounters
        # Calculate maximum theoretically possible

        # All possible team pairs (choosing 2 from 4): C(4,2) = 6
        # AB, AC, AD, BC, BD, CD
        # Each can be in 2 positions: AB vs CD, AC vs BD, etc.
        # But need to account for team vs team uniqueness

        max_rounds = 15  # Test high number
        schedule = self.scheduler.generate_schedule(max_rounds)

        # Should not have exact repetitions
        encounters = set(schedule)
        assert len(encounters) == len(schedule)

    def test_impossible_constraints(self):
        """Test when constraints cannot be satisfied"""
        # This might happen with very specific constraint combinations
        # For now, test that solver handles gracefully

        try:
            # Try to generate schedule with impossible constraints
            # (This test may evolve as we understand constraint limits)
            schedule = self.scheduler.generate_schedule(100)  # Very high number
            # Should either succeed or fail gracefully
            assert isinstance(schedule, list)
        except ValueError as e:
            # Should raise informative error about inability to find solution
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in
                      ["impossible", "unsatisfiable", "no valid", "not found"]), f"Unexpected error message: {e}"

    def test_schedule_properties(self):
        """Test general properties any valid schedule must have"""
        rounds = 5
        schedule = self.scheduler.generate_schedule(rounds)

        assert len(schedule) == rounds

        for i, (team1, team2) in enumerate(schedule):
            # Each round has exactly 4 players
            all_players = set(team1 + team2)
            assert len(all_players) == 4, f"Round {i+1} doesn't have 4 players"
            assert all_players == set(self.players), f"Round {i+1} has wrong players"

            # Teams are properly formed
            assert len(team1) == 2 and len(team2) == 2
            assert len(set(team1)) == 2 and len(set(team2)) == 2  # No duplicates within team

    def test_position_optimization_score(self):
        """Test position switching scoring"""
        schedule = [
            ("AB", "CD"),  # A=D, B=A, C=D, D=A
            ("CA", "DB"),  # C=D, A=A, D=D, B=A  (A: D->A ✓, B: A->A ✗, C: D->D ✗, D: A->D ✓)
        ]

        score = self.scheduler.calculate_position_switching_score(schedule)
        # Should reward position switches, penalize same positions
        assert score >= 0  # Non-negative score

    def test_team_mixing_score(self):
        """Test team mixing scoring based on 5-tier hierarchy"""
        schedule = [
            ("AB", "CD"),  # Round 1
            ("CA", "DB"),  # Round 2: Perfect mixing (all change teammates)
        ]

        score = self.scheduler.calculate_team_mixing_score(schedule)
        assert score >= 0  # Non-negative score

        # Perfect mixing should score higher than minimal mixing
        poor_schedule = [
            ("AB", "CD"),  # Round 1
            ("BA", "CD"),  # Round 2: Minimal change (only A,B switch positions)
        ]

        poor_score = self.scheduler.calculate_team_mixing_score(poor_schedule)
        assert score > poor_score  # Perfect mixing should score higher


if __name__ == "__main__":
    pytest.main([__file__])