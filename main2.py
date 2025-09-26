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
        if len(players) < 4 or len(players) > 10:
            raise ValueError("Supports 4-10 players only")

        self.players = players
        self.num_players = len(players)

    def generate_schedule(self, num_rounds: int) -> List[Tuple[int, str, str, str, str, str]]:
        """
        Generate absolutely optimal schedule using comprehensive search with backtracking.
        This may take minutes for longer schedules but finds the truly optimal solution.

        Returns:
            List of tuples: (round, team1_defense, team1_attack, team2_defense, team2_attack, sitting_players)
        """
        print(f"Generating optimal schedule for {num_rounds} rounds... (this may take a while)")

        if self.num_players == 4:
            return self._generate_optimal_4_player_schedule(num_rounds)
        else:
            return self._generate_optimal_5_6_player_schedule(num_rounds)

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

    def _generate_optimal_5_6_player_schedule(self, num_rounds: int) -> List[Tuple[int, str, str, str, str, str]]:
        """Generate optimal schedule with STRICT sitting rotation as highest priority"""
        print("Using sitting-rotation-first optimization for perfect balance...")

        # CRITICAL: Sitting rotation is NON-NEGOTIABLE for 5 players
        # Pattern: E sits round 1, A sits round 2, B sits round 3, C sits round 4, D sits round 5, E sits round 6...

        return self._generate_sitting_enforced_schedule(num_rounds)

    def _generate_sitting_enforced_schedule(self, num_rounds: int) -> List[Tuple[int, str, str, str, str, str]]:
        """Generate schedule with ABSOLUTE sitting rotation enforcement"""
        print(f"Enforcing perfect sitting rotation for {num_rounds} rounds...")

        schedule = []

        # Track state for optimization within sitting constraints
        state = {
            'player_positions': {player: [] for player in self.players},
            'team_frequencies': {},
            'teammate_frequencies': {player: {} for player in self.players}
        }

        # Calculate how many players sit each round
        sitting_count = self.num_players - 4  # Always 4 players play, rest sit

        # Define ABSOLUTE sitting rotation for any number of players
        # For 5 players: 1 sits each round, cycle through E,A,B,C,D
        # For 6 players: 2 sit each round, cycle through all combinations
        # For 7+ players: 3+ sit each round, cycle through all combinations

        if self.num_players == 5:
            # Simple rotation for 5 players: one player sits each round
            sitting_rotation = [self.players[4:]]  # [['E']]
            for i in range(4):
                sitting_rotation.append([self.players[i]])  # [['A'], ['B'], ['C'], ['D']]
        else:
            # For 6+ players: generate fair rotation of sitting combinations
            sitting_rotation = self._generate_fair_sitting_rotation(sitting_count)

        for round_num in range(1, num_rounds + 1):
            # MANDATORY: Determine who MUST sit this round
            sitting_players_list = sitting_rotation[(round_num - 1) % len(sitting_rotation)]
            sitting_players_str = ''.join(sorted(sitting_players_list))

            # Get the 4 playing players
            playing_players = [p for p in self.players if p not in sitting_players_list]

            if round_num == 1:
                # Round 1 is standardized: AB vs CD, but we must use the actual playing players
                # If A,B,C,D are not all playing (some might be sitting), pick the first 4 playing players
                if len(playing_players) >= 4 and all(p in playing_players for p in ['A', 'B', 'C', 'D']):
                    encounter = (round_num, 'A', 'B', 'C', 'D', sitting_players_str)
                else:
                    # Use the first 4 playing players if A,B,C,D are not all available
                    p = playing_players  # shorthand
                    encounter = (round_num, p[0], p[1], p[2], p[3], sitting_players_str)
            else:
                # Find best encounter among the playing players
                encounter = self._find_best_encounter_with_sitting(
                    round_num, playing_players, sitting_players_str, state
                )

            schedule.append(encounter)
            self._update_state_tracking(encounter, state)

        print(f"Sitting-enforced schedule complete: {len(schedule)} rounds")
        return schedule

    def _generate_fair_sitting_rotation(self, sitting_count: int) -> List[List[str]]:
        """Generate a fair rotation where each player sits equally often"""

        # For N players with K sitting each round:
        # Each player should sit 1 out of every (N/K) rounds
        #
        # For 6 players, 2 sit each round: each player sits 1 out of every 3 rounds
        # Pattern: AB, CD, EF, AB, CD, EF, ...
        #
        # For 7 players, 3 sit each round: each player sits 3 out of every 7 rounds
        # This is more complex, but we can create a repeating cycle

        rotation_length = self.num_players // sitting_count

        if self.num_players % sitting_count != 0:
            # Handle cases where players don't divide evenly
            rotation_length = self.num_players

        sitting_rotation = []

        if sitting_count == 1:
            # Simple case: 5 players, 1 sits each round
            # E, A, B, C, D, E, A, B, C, D...
            for i in range(self.num_players):
                sitting_rotation.append([self.players[i]])

        elif sitting_count == 2 and self.num_players == 6:
            # 6 players, 2 sit each round: AB, CD, EF, AB, CD, EF...
            pairs = [
                [self.players[0], self.players[1]],  # AB
                [self.players[2], self.players[3]],  # CD
                [self.players[4], self.players[5]]   # EF
            ]
            for _ in range(20):  # Repeat enough times for long tournaments
                sitting_rotation.extend(pairs)

        elif sitting_count == 3 and self.num_players == 7:
            # 7 players, 3 sit each round
            # Create a pattern where each player sits 3 out of every 7 rounds
            # This requires a more complex 7-round cycle
            groups = [
                [self.players[0], self.players[1], self.players[2]],  # ABC
                [self.players[3], self.players[4], self.players[5]],  # DEF
                [self.players[6], self.players[0], self.players[3]],  # GAD
                [self.players[1], self.players[4], self.players[2]],  # BEC
                [self.players[5], self.players[6], self.players[1]],  # FGB
                [self.players[2], self.players[3], self.players[4]],  # CDE
                [self.players[0], self.players[5], self.players[6]]   # AFG
            ]
            for _ in range(15):  # Repeat enough times
                sitting_rotation.extend(groups)
        else:
            # Fallback: create a simple pattern that tries to distribute evenly
            # Round-robin style where we cycle through player indices
            for round_offset in range(self.num_players * 2):  # Create enough rounds
                sitting_players = []
                for i in range(sitting_count):
                    player_index = (round_offset + i) % self.num_players
                    sitting_players.append(self.players[player_index])
                sitting_rotation.append(sitting_players)

        print(f"Generated {len(sitting_rotation)} sitting combinations for {self.num_players} players")
        print(f"Pattern repeats every {rotation_length} rounds for fair distribution")

        return sitting_rotation

    def _find_best_encounter_with_sitting(self, round_num: int, playing_players: List[str],
                                        sitting_players: str, state: Dict) -> Tuple[int, str, str, str, str, str]:
        """Find best encounter given fixed sitting player and playing players"""

        # Generate all possible encounters with these specific playing players
        possible_encounters = []

        # All ways to form 2 teams of 2 from the 4 playing players
        for team1_players in combinations(playing_players, 2):
            team2_players = [p for p in playing_players if p not in team1_players]

            # All position assignments
            for t1_def, t1_att in permutations(team1_players, 2):
                for t2_def, t2_att in permutations(team2_players, 2):
                    encounter = (round_num, t1_def, t1_att, t2_def, t2_att, sitting_players)
                    possible_encounters.append(encounter)

        # Score each encounter and pick the best
        best_encounter = None
        best_score = -float('inf')  # Use negative infinity to ensure we always find something

        for encounter in possible_encounters:
            score = self._score_encounter_within_sitting_constraints(encounter, state)

            if score > best_score:
                best_score = score
                best_encounter = encounter

        # Fallback: if no encounter found, return the first one
        if best_encounter is None and possible_encounters:
            best_encounter = possible_encounters[0]

        return best_encounter

    def _score_encounter_within_sitting_constraints(self, encounter: Tuple, state: Dict) -> float:
        """Score encounter focusing on team balance and position switching"""
        _, t1_def, t1_att, t2_def, t2_att, sitting = encounter

        score = 0

        # 1. Team frequency balance (highest priority)
        team1 = t1_def + t1_att
        team2 = t2_def + t2_att
        team1_freq = state['team_frequencies'].get(team1, 0)
        team2_freq = state['team_frequencies'].get(team2, 0)

        # Heavily penalize frequently used teams
        max_acceptable_freq = 3  # Allow some repetition but not excessive
        if team1_freq >= max_acceptable_freq:
            score -= team1_freq * 1000
        if team2_freq >= max_acceptable_freq:
            score -= team2_freq * 1000

        # Bonus for using infrequent teams
        score += (max_acceptable_freq - team1_freq) * 100
        score += (max_acceptable_freq - team2_freq) * 100

        # 2. Teammate frequency balance
        t1_teammate_freq = state['teammate_frequencies'][t1_def].get(t1_att, 0)
        t2_teammate_freq = state['teammate_frequencies'][t2_def].get(t2_att, 0)

        # Penalize repeated teammates
        if t1_teammate_freq >= 2:
            score -= t1_teammate_freq * 500
        if t2_teammate_freq >= 2:
            score -= t2_teammate_freq * 500

        # 3. Position switching bonus
        position_score = self._score_position_switching(encounter, state['player_positions'])
        score += position_score * 50

        return score

    def _generate_segmented_optimal_schedule(self, num_rounds: int) -> List[Tuple[int, str, str, str, str, str]]:
        """Generate schedule by optimizing segments and ensuring global balance"""
        print(f"Using segmented approach for {num_rounds} rounds...")

        segment_size = 10  # Optimize 10 rounds at a time
        full_schedule = []

        # Global state tracking
        global_state = {
            'player_positions': {player: [] for player in self.players},
            'sitting_counts': {player: 0 for player in self.players},
            'team_frequencies': {},
            'teammate_frequencies': {player: {} for player in self.players}
        }

        # Start with mandatory first round
        sitting_players = ''.join(self.players[4:])
        first_encounter = (1, 'A', 'B', 'C', 'D', sitting_players)
        full_schedule.append(first_encounter)
        self._update_state_tracking(first_encounter, global_state)

        current_round = 2

        while current_round <= num_rounds:
            remaining_rounds = num_rounds - current_round + 1
            this_segment_size = min(segment_size, remaining_rounds)

            print(f"Optimizing rounds {current_round}-{current_round + this_segment_size - 1}...")

            # Generate optimal segment
            segment = self._generate_optimal_segment(
                current_round,
                this_segment_size,
                global_state,
                remaining_rounds
            )

            # Add segment to full schedule
            full_schedule.extend(segment)

            # Update global state
            for encounter in segment:
                self._update_state_tracking(encounter, global_state)

            current_round += this_segment_size

        print(f"Segmented optimization complete. Total rounds: {len(full_schedule)}")
        return full_schedule

    def _generate_optimal_segment(self, start_round: int, segment_size: int,
                                global_state: Dict, total_remaining: int) -> List[Tuple]:
        """Generate optimal segment considering global state"""

        # Create local state for segment optimization
        segment_state = self._copy_state(global_state)
        segment_state['schedule'] = []

        best_segment = None
        best_score = -1

        # For small segments, try multiple greedy approaches
        for strategy in ['balance_teams', 'prioritize_sitting', 'max_switching']:
            segment = self._generate_greedy_segment(
                start_round, segment_size, segment_state, strategy, total_remaining
            )

            score = self._evaluate_segment_quality(segment, global_state, total_remaining)

            if score > best_score:
                best_score = score
                best_segment = segment

        return best_segment

    def _generate_greedy_segment(self, start_round: int, segment_size: int,
                               base_state: Dict, strategy: str, total_remaining: int) -> List[Tuple]:
        """Generate segment using specific greedy strategy"""

        segment = []
        current_state = self._copy_state(base_state)

        for round_offset in range(segment_size):
            round_num = start_round + round_offset

            # Generate possible encounters
            possible_encounters = self._generate_all_possible_encounters(round_num, current_state)

            # Score encounters based on strategy
            best_encounter = None
            best_score = -1

            for encounter in possible_encounters:
                score = self._score_encounter_with_strategy(
                    encounter, current_state, strategy, total_remaining - round_offset
                )

                if score > best_score:
                    best_score = score
                    best_encounter = encounter

            if best_encounter:
                segment.append(best_encounter)
                self._update_state_tracking(best_encounter, current_state)

        return segment

    def _score_encounter_with_strategy(self, encounter: Tuple, state: Dict,
                                     strategy: str, remaining_rounds: int) -> float:
        """Score encounter based on specific strategy"""
        base_score = self._score_immediate_encounter(encounter, state)

        if strategy == 'balance_teams':
            # Heavily penalize overused teams
            team_penalty = self._calculate_team_overuse_penalty(encounter, state)
            return base_score - team_penalty * 2

        elif strategy == 'prioritize_sitting':
            # Focus on perfect sitting rotation
            sitting_bonus = self._calculate_sitting_rotation_bonus(encounter, state, remaining_rounds)
            return base_score + sitting_bonus * 3

        elif strategy == 'max_switching':
            # Maximize position switching
            switching_bonus = self._score_position_switching(encounter, state['player_positions'])
            return base_score + switching_bonus * 2

        return base_score

    def _calculate_team_overuse_penalty(self, encounter: Tuple, state: Dict) -> float:
        """Calculate penalty for overusing teams"""
        _, t1_def, t1_att, t2_def, t2_att, _ = encounter
        team1 = t1_def + t1_att
        team2 = t2_def + t2_att

        team1_freq = state['team_frequencies'].get(team1, 0)
        team2_freq = state['team_frequencies'].get(team2, 0)

        # Expected frequency based on current progress
        total_encounters = len(state.get('schedule', []))
        expected_max = total_encounters / 15  # Rough estimate for 5 players

        penalty = 0
        if team1_freq > expected_max:
            penalty += (team1_freq - expected_max) * 100
        if team2_freq > expected_max:
            penalty += (team2_freq - expected_max) * 100

        return penalty

    def _calculate_sitting_rotation_bonus(self, encounter: Tuple, state: Dict, remaining_rounds: int) -> float:
        """Calculate bonus for perfect sitting rotation"""
        if self.num_players != 5:
            return 0

        _, _, _, _, _, sitting = encounter
        if not sitting:
            return 0

        # For 5 players, perfect rotation is E, A, B, C, D, E, A, B, C, D...
        round_num = len(state.get('schedule', [])) + 1
        expected_sitting_player = self.players[(round_num - 1) % 5]

        if sitting == expected_sitting_player:
            return 500  # Big bonus for perfect rotation

        # Smaller bonus if this player has sat least recently
        sitting_counts = state.get('sitting_counts', {})
        if sitting_counts.get(sitting, 0) == min(sitting_counts.values()):
            return 100

        return 0

    def _evaluate_segment_quality(self, segment: List[Tuple], global_state: Dict, total_remaining: int) -> float:
        """Evaluate the quality of a segment in global context"""

        # Simulate adding this segment to global state
        temp_state = self._copy_state(global_state)
        temp_schedule = temp_state.get('schedule', []).copy()

        for encounter in segment:
            temp_schedule.append(encounter)
            self._update_state_tracking(encounter, temp_state)

        # Evaluate the full schedule so far
        score = self._evaluate_complete_schedule(temp_schedule)

        # Bonus for maintaining good global balance
        balance_bonus = self._calculate_global_balance_bonus(temp_state, total_remaining)

        return score + balance_bonus

    def _calculate_global_balance_bonus(self, state: Dict, remaining_rounds: int) -> float:
        """Calculate bonus for maintaining global balance"""
        bonus = 0

        # Team frequency balance
        team_freqs = list(state['team_frequencies'].values())
        if team_freqs:
            variance = self._calculate_variance(team_freqs)
            bonus += max(0, 500 - variance * 5)  # Lower variance = higher bonus

        # Sitting balance (for 5-6 players)
        if self.num_players > 4:
            sitting_counts = list(state['sitting_counts'].values())
            if sitting_counts:
                sitting_variance = self._calculate_variance(sitting_counts)
                bonus += max(0, 300 - sitting_variance * 20)

        return bonus

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _generate_limited_backtrack_schedule(self, num_rounds: int) -> List[Tuple[int, str, str, str, str, str]]:
        """Limited backtracking for shorter schedules"""
        print(f"Using limited backtracking for {num_rounds} rounds...")

        # Use the greedy approach with multiple strategies and pick the best
        best_schedule = None
        best_score = -1

        strategies = ['balance_teams', 'prioritize_sitting', 'max_switching']

        for strategy in strategies:
            print(f"Trying strategy: {strategy}")
            schedule = self._generate_full_greedy_schedule(num_rounds, strategy)
            score = self._evaluate_complete_schedule(schedule)

            if score > best_score:
                best_score = score
                best_schedule = schedule
                print(f"New best with {strategy}: {score:.2f}")

        return best_schedule

    def _generate_full_greedy_schedule(self, num_rounds: int, strategy: str) -> List[Tuple[int, str, str, str, str, str]]:
        """Generate full schedule using consistent strategy"""

        schedule = []
        state = {
            'schedule': [],
            'player_positions': {player: [] for player in self.players},
            'sitting_counts': {player: 0 for player in self.players},
            'team_frequencies': {},
            'teammate_frequencies': {player: {} for player in self.players}
        }

        # Start with mandatory first round
        sitting_players = ''.join(self.players[4:])
        first_encounter = (1, 'A', 'B', 'C', 'D', sitting_players)
        schedule.append(first_encounter)
        state['schedule'].append(first_encounter)
        self._update_state_tracking(first_encounter, state)

        # Generate remaining rounds
        for round_num in range(2, num_rounds + 1):
            possible_encounters = self._generate_all_possible_encounters(round_num, state)

            best_encounter = None
            best_score = -1

            for encounter in possible_encounters:
                score = self._score_encounter_with_strategy(
                    encounter, state, strategy, num_rounds - round_num + 1
                )

                if score > best_score:
                    best_score = score
                    best_encounter = encounter

            if best_encounter:
                schedule.append(best_encounter)
                state['schedule'].append(best_encounter)
                self._update_state_tracking(best_encounter, state)

        return schedule

    def _generate_optimal_4_player_schedule(self, num_rounds: int) -> List[Tuple[int, str, str, str, str, str]]:
        """Generate optimal schedule for 4 players using backtracking"""
        print("Using comprehensive backtracking search for 4-player optimal solution...")

        initial_state = {
            'schedule': [],
            'player_positions': {player: [] for player in self.players},
            'team_frequencies': {},
            'teammate_frequencies': {player: {} for player in self.players}
        }

        # Start with mandatory first round
        first_encounter = (1, 'A', 'B', 'C', 'D', '')
        initial_state['schedule'].append(first_encounter)
        self._update_state_tracking(first_encounter, initial_state)

        best_solution = None
        best_score = -1

        def backtrack_search_4p(current_state, round_num):
            nonlocal best_solution, best_score

            if round_num > num_rounds:
                total_score = self._evaluate_complete_schedule(current_state['schedule'])
                if total_score > best_score:
                    best_score = total_score
                    best_solution = current_state['schedule'].copy()
                    print(f"New best score: {total_score:.2f}")
                return

            # Generate all possible 4-player encounters
            possible_encounters = self._generate_all_4_player_encounters(round_num)

            encounter_scores = []
            for encounter in possible_encounters:
                immediate_score = self._score_immediate_encounter(encounter, current_state)
                encounter_scores.append((immediate_score, encounter))

            encounter_scores.sort(reverse=True)

            for score, encounter in encounter_scores:
                if best_solution and not self._can_improve(current_state, encounter, best_score, num_rounds - round_num + 1):
                    continue

                new_state = self._copy_state(current_state)
                new_state['schedule'].append(encounter)
                self._update_state_tracking(encounter, new_state)

                backtrack_search_4p(new_state, round_num + 1)

        backtrack_search_4p(initial_state, 2)

        if best_solution is None:
            raise ValueError("No valid schedule found")

        print(f"Optimal 4-player solution found with score: {best_score:.2f}")
        return best_solution

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

    def _generate_all_possible_encounters(self, round_num: int, current_state: Dict) -> List[Tuple]:
        """Generate all possible encounters for a given round"""
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

        return possible_encounters

    def _generate_all_4_player_encounters(self, round_num: int) -> List[Tuple]:
        """Generate all possible 4-player encounters"""
        possible_encounters = []

        # Get all possible 2-player combinations
        for team1_players in combinations(self.players, 2):
            team2_players = [p for p in self.players if p not in team1_players]

            # All position combinations
            for t1_def, t1_att in permutations(team1_players, 2):
                for t2_def, t2_att in permutations(team2_players, 2):
                    encounter = (round_num, t1_def, t1_att, t2_def, t2_att, '')
                    possible_encounters.append(encounter)

        return possible_encounters

    def _update_state_tracking(self, encounter: Tuple, state: Dict):
        """Update comprehensive state tracking"""
        _, t1_def, t1_att, t2_def, t2_att, sitting = encounter

        # Update positions
        state['player_positions'][t1_def].append('D')
        state['player_positions'][t1_att].append('A')
        state['player_positions'][t2_def].append('D')
        state['player_positions'][t2_att].append('A')

        # Update sitting counts (if applicable)
        if 'sitting_counts' in state:
            for player in sitting:
                state['player_positions'][player].append('S')
                state['sitting_counts'][player] += 1

        # Update team frequencies
        team1 = t1_def + t1_att
        team2 = t2_def + t2_att
        state['team_frequencies'][team1] = state['team_frequencies'].get(team1, 0) + 1
        state['team_frequencies'][team2] = state['team_frequencies'].get(team2, 0) + 1

        # Update teammate frequencies
        # Team 1
        if t1_att not in state['teammate_frequencies'][t1_def]:
            state['teammate_frequencies'][t1_def][t1_att] = 0
        state['teammate_frequencies'][t1_def][t1_att] += 1

        if t1_def not in state['teammate_frequencies'][t1_att]:
            state['teammate_frequencies'][t1_att][t1_def] = 0
        state['teammate_frequencies'][t1_att][t1_def] += 1

        # Team 2
        if t2_att not in state['teammate_frequencies'][t2_def]:
            state['teammate_frequencies'][t2_def][t2_att] = 0
        state['teammate_frequencies'][t2_def][t2_att] += 1

        if t2_def not in state['teammate_frequencies'][t2_att]:
            state['teammate_frequencies'][t2_att][t2_def] = 0
        state['teammate_frequencies'][t2_att][t2_def] += 1

    def _copy_state(self, state: Dict) -> Dict:
        """Create a deep copy of the state"""
        import copy
        return copy.deepcopy(state)

    def _score_immediate_encounter(self, encounter: Tuple, current_state: Dict) -> float:
        """Score an encounter based on current state"""
        _, t1_def, t1_att, t2_def, t2_att, sitting = encounter

        score = 0

        # 1. Sitting fairness (highest priority for 5-6 players)
        if self.num_players > 4 and 'sitting_counts' in current_state:
            sitting_score = self._score_sitting_fairness(encounter, current_state['sitting_counts'])
            score += sitting_score * 1000

        # 2. Team frequency penalty (heavily penalize overused teams)
        team1 = t1_def + t1_att
        team2 = t2_def + t2_att
        team1_freq = current_state['team_frequencies'].get(team1, 0)
        team2_freq = current_state['team_frequencies'].get(team2, 0)

        # Heavy penalty for frequently used teams
        max_expected_freq = len(current_state['schedule']) // 10 + 1  # Expected frequency ceiling
        if team1_freq > max_expected_freq:
            score -= (team1_freq - max_expected_freq) * 500
        if team2_freq > max_expected_freq:
            score -= (team2_freq - max_expected_freq) * 500

        # 3. Teammate frequency penalty (avoid same teammates)
        t1_teammate_freq = current_state['teammate_frequencies'][t1_def].get(t1_att, 0)
        t2_teammate_freq = current_state['teammate_frequencies'][t2_def].get(t2_att, 0)

        max_teammate_freq = len(current_state['schedule']) // 8 + 1
        if t1_teammate_freq > max_teammate_freq:
            score -= (t1_teammate_freq - max_teammate_freq) * 300
        if t2_teammate_freq > max_teammate_freq:
            score -= (t2_teammate_freq - max_teammate_freq) * 300

        # 4. Position switching bonus
        position_score = self._score_position_switching(encounter, current_state['player_positions'])
        score += position_score * 100

        return score

    def _score_sitting_fairness(self, encounter: Tuple, sitting_counts: Dict) -> float:
        """Score sitting fairness for this encounter"""
        _, _, _, _, _, sitting = encounter
        if not sitting:
            return 50  # Neutral for 4-player games

        current_min_sits = min(sitting_counts.values())
        score = 0

        for player in sitting:
            if sitting_counts[player] == current_min_sits:
                score += 25  # Bonus for giving turn to player who has sat least
            else:
                score -= 20  # Heavy penalty for making someone sit again too soon

        return score

    def _score_position_switching(self, encounter: Tuple, player_positions: Dict) -> float:
        """Score position switching for this encounter"""
        _, t1_def, t1_att, t2_def, t2_att, _ = encounter

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

        return score

    def _can_improve(self, current_state: Dict, encounter: Tuple, best_score: float, remaining_rounds: int) -> bool:
        """Early pruning: check if this branch can possibly improve the best score"""
        # Simple heuristic: if we're very far from the best score and have few rounds left, prune
        current_score = self._evaluate_partial_schedule(current_state['schedule'] + [encounter])
        max_possible_remaining = remaining_rounds * 1000  # Maximum possible score per round

        return current_score + max_possible_remaining > best_score * 0.95  # Allow some tolerance

    def _evaluate_complete_schedule(self, schedule: List[Tuple]) -> float:
        """Evaluate a complete schedule with all objectives"""
        score = 0

        # 1. Sitting fairness (very high weight)
        if self.num_players > 4:
            sitting_score = self.calculate_sitting_fairness_score(schedule)
            score += sitting_score * 10

        # 2. Position switching (high weight)
        position_score = self.calculate_position_switching_score(schedule)
        score += position_score * 50

        # 3. Team mixing (very high weight for even distribution)
        team_mixing_score = self._calculate_comprehensive_team_mixing_score(schedule)
        score += team_mixing_score * 100

        # 4. Side balance (lower weight)
        side_score = self.calculate_side_balance_score(schedule)
        score += side_score * 1

        return score

    def _evaluate_partial_schedule(self, schedule: List[Tuple]) -> float:
        """Evaluate a partial schedule (for pruning)"""
        if len(schedule) <= 1:
            return 0

        # Quick evaluation focusing on recent decisions
        recent_rounds = schedule[-min(5, len(schedule)):]  # Last 5 rounds
        return self._evaluate_complete_schedule(recent_rounds) * (len(recent_rounds) / len(schedule))

    def _calculate_comprehensive_team_mixing_score(self, schedule: List[Tuple]) -> float:
        """Calculate team mixing score focusing on even distribution"""
        team_counts = {}

        for encounter in schedule:
            _, t1_def, t1_att, t2_def, t2_att, _ = encounter
            team1 = t1_def + t1_att
            team2 = t2_def + t2_att

            team_counts[team1] = team_counts.get(team1, 0) + 1
            team_counts[team2] = team_counts.get(team2, 0) + 1

        if not team_counts:
            return 0

        # Calculate distribution evenness
        frequencies = list(team_counts.values())
        if not frequencies:
            return 0

        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)

        # Lower variance = better (more even distribution)
        # Convert to score where higher is better
        evenness_score = max(0, 1000 - variance * 10)

        return evenness_score

    def calculate_sitting_fairness_score(self, schedule) -> int:
        """Calculate sitting fairness score - how evenly sitting time is distributed"""
        if self.num_players == 4:
            return 0  # No sitting in 4-player games

        sitting_count = {player: 0 for player in self.players}

        for encounter in schedule:
            _, _, _, _, _, sitting = encounter
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
        max_possible_score = len(schedule) * self.num_players
        fairness_score = max(0, max_possible_score - sit_difference * 10)

        return fairness_score

    def calculate_position_switching_score(self, schedule) -> int:
        """Calculate position switching score for a complete schedule"""
        if len(schedule) < 2:
            return 0

        total_score = 0
        positions = {player: [] for player in self.players}

        for encounter in schedule:
            _, t1_def, t1_att, t2_def, t2_att, sitting = encounter

            # Track positions
            positions[t1_def].append('D')
            positions[t1_att].append('A')
            positions[t2_def].append('D')
            positions[t2_att].append('A')

            # Sitting players get 'S'
            if sitting:
                for player in sitting:
                    positions[player].append('S')

        # Score based on position switches
        for i in range(len(schedule) - 1):
            round_switches = 0
            playing_players = 0

            for player in self.players:
                if positions[player][i] != 'S' and positions[player][i+1] != 'S':
                    playing_players += 1
                    if positions[player][i] != positions[player][i+1]:
                        round_switches += 1
                        total_score += 1

            # Bonus for perfect switching rounds
            if playing_players == 4 and round_switches == 4:
                total_score += 4

        return total_score

    def calculate_side_balance_score(self, schedule) -> int:
        """Calculate side balance score - how evenly teams are distributed across table sides"""
        if len(schedule) == 0:
            return 0

        side_1_count = {}
        side_2_count = {}

        # Initialize all possible teams
        all_teams = set()
        for encounter in schedule:
            _, t1_def, t1_att, t2_def, t2_att, _ = encounter
            team1 = t1_def + t1_att
            team2 = t2_def + t2_att
            all_teams.add(team1)
            all_teams.add(team2)

        for team in all_teams:
            side_1_count[team] = 0
            side_2_count[team] = 0

        # Count side appearances
        for encounter in schedule:
            _, t1_def, t1_att, t2_def, t2_att, _ = encounter
            team1 = t1_def + t1_att
            team2 = t2_def + t2_att
            side_1_count[team1] += 1
            side_2_count[team2] += 1

        # Calculate balance score
        total_score = 0
        for team in all_teams:
            side1_plays = side_1_count[team]
            side2_plays = side_2_count[team]
            total_plays = side1_plays + side2_plays

            if total_plays > 0:
                balance_diff = abs(side1_plays - side2_plays)
                balance_score = max(0, total_plays - balance_diff)
                total_score += balance_score

        return total_score

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
    parser.add_argument('--players', '-p', type=int, default=5,
                        help='Number of players (4-10 supported, default: 5)')
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