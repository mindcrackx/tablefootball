"""
Example usage of the Table Football Scheduler
"""
from main import TableFootballScheduler


def main():
    """Demonstrate different scheduling scenarios"""
    players = ['A', 'B', 'C', 'D']
    scheduler = TableFootballScheduler(players)

    print("🏓 Table Football Tournament Scheduler Examples")
    print("=" * 60)

    # Example 1: Short tournament (3 rounds)
    print("\n📋 Example 1: Short Tournament (3 rounds)")
    print("-" * 40)
    try:
        schedule = scheduler.generate_schedule(3)
        scheduler.print_schedule(schedule)
    except ValueError as e:
        print(f"Error: {e}")

    # Example 2: Medium tournament (7 rounds)
    print("\n📋 Example 2: Medium Tournament (7 rounds)")
    print("-" * 40)
    try:
        schedule = scheduler.generate_schedule(7)
        scheduler.print_schedule(schedule)
    except ValueError as e:
        print(f"Error: {e}")

    # Example 3: Test the known good schedule from PROBLEM.md
    print("\n📋 Example 3: Verify Known Good Schedule from PROBLEM.md")
    print("-" * 40)
    known_schedule = [
        ("AB", "CD"),  # Round 1
        ("CA", "DB"),  # Round 2
        ("AD", "BC")   # Round 3
    ]

    print("Known Schedule:")
    scheduler.print_schedule(known_schedule)

    # Example 4: Show scheduling statistics
    print("\n📊 Example 4: Scheduling Statistics")
    print("-" * 40)

    total_possible_encounters = len(scheduler.all_encounters)
    print(f"Total possible unique encounters: {total_possible_encounters}")

    print("\nUnique encounters available:")
    for i, encounter in enumerate(scheduler.all_encounters[:10]):  # Show first 10
        print(f"  {i+1:2d}. {encounter[0]} vs {encounter[1]}")
    if total_possible_encounters > 10:
        print(f"  ... and {total_possible_encounters - 10} more")

    # Example 5: Compare different schedule lengths
    print("\n📈 Example 5: Schedule Quality vs Length")
    print("-" * 40)
    for rounds in [1, 3, 5, 8, 10]:
        try:
            schedule = scheduler.generate_schedule(rounds)
            pos_score = scheduler.calculate_position_switching_score(schedule)
            mix_score = scheduler.calculate_team_mixing_score(schedule)
            total_score = mix_score * 5 + pos_score * 3

            print(f"  {rounds:2d} rounds: Pos={pos_score:2d}, Mix={mix_score:2d}, Total={total_score:3d}")
        except ValueError:
            print(f"  {rounds:2d} rounds: Not possible (too many rounds)")


if __name__ == "__main__":
    main()