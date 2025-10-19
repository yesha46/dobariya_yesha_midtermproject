import pandas as pd
import itertools
import os
import time

# Get Available CSV Files
def get_csv_files(folder_path):
    return [f for f in os.listdir(folder_path)
            if f.endswith(".csv") and not f.startswith("frequent_itemsets_") and not f.startswith("rules_")]

# Load Transactions
def load_transactions(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp1252', on_bad_lines='skip')

    df.columns = [c.strip().lower() for c in df.columns]
    if 'items' not in df.columns:
        raise ValueError("CSV must contain a column named 'Items'")

    transactions = df['items'].apply(lambda x: [i.strip() for i in str(x).split(',')])
    return transactions.tolist()

# Input validation for support/confidence
def get_valid_float(prompt):
    while True:
        try:
            val = float(input(prompt))
            if 0 < val <= 1:
                return val
            print("Please enter a number between 0 and 1 (exclusive of 0).")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Get Frequent Itemsets
def get_frequent_itemsets(transactions, min_support):
    items = sorted(set(itertools.chain.from_iterable(transactions)))
    all_frequent = {}
    k = 1
    num_transactions = len(transactions)
    start = time.perf_counter()

    while True:
        candidates = list(itertools.combinations(items, k))
        frequent = {}
        for c in candidates:
            count = sum(1 for t in transactions if set(c).issubset(set(t)))
            support = count / num_transactions
            if support >= min_support:
                frequent[c] = {"support": support, "count": count}
        if not frequent:
            break
        all_frequent[k] = frequent
        k += 1

    end = time.perf_counter()
    return all_frequent, end - start

# Generate Association Rules
def generate_rules(frequent_itemsets, min_confidence, transactions):
    rules = []
    num_transactions = len(transactions)
    start = time.perf_counter()
    flat_itemsets = {item: val for level in frequent_itemsets.values() for item, val in level.items()}

    for itemset, metrics in flat_itemsets.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for lhs in itertools.combinations(itemset, i):
                rhs = tuple(sorted(set(itemset) - set(lhs)))
                lhs_count = sum(1 for t in transactions if set(lhs).issubset(set(t)))
                if lhs_count == 0:
                    continue
                confidence = metrics["support"] / (lhs_count / num_transactions)
                if confidence >= min_confidence:
                    rules.append({
                        "lhs": lhs,
                        "rhs": rhs,
                        "support": metrics["support"],
                        "confidence": confidence,
                        "count": metrics["count"]
                    })

    end = time.perf_counter()
    return rules, end - start

# Main Function
def main():
    # Path to data folder relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    folder_path = os.path.join(script_dir, "..", "data")    
    csv_files = get_csv_files(folder_path)

    if not csv_files:
        print("No CSV files found in the data folder!")
        return

    print("Available datasets:")
    for i, f in enumerate(csv_files, start=1):
        print(f"{i}. {f}")

    try:
        choice = int(input("\nSelect a dataset by number: "))
        if choice < 1 or choice > len(csv_files):
            raise ValueError
    except ValueError:
        print("Invalid input. Please enter a valid dataset number.")
        return

    dataset = csv_files[choice - 1]
    file_path = os.path.join(folder_path, dataset)

    min_support = get_valid_float("Enter minimum support (e.g., 0.2 for 20%): ")
    min_confidence = get_valid_float("Enter minimum confidence (e.g., 0.6 for 60%): ")

    print(f"\nRunning Brute Force Algorithm on '{dataset}' ...")
    transactions = load_transactions(file_path)

    start_total = time.perf_counter()
    frequent_itemsets, fi_time = get_frequent_itemsets(transactions, min_support)
    rules, rules_time = generate_rules(frequent_itemsets, min_confidence, transactions)
    end_total = time.perf_counter()
    total_time = end_total - start_total

    # Print Frequent Itemsets
    for k in sorted(frequent_itemsets.keys()):
        print(f"\n===== {k}-Frequent Itemsets =====")
        for itemset, metrics in sorted(frequent_itemsets[k].items(), key=lambda x: -x[1]["support"]):
            print(f"{itemset}: support={metrics['support']:.2f}, count={metrics['count']}")
        print()

    # Print Association Rules
    print("\n===== Association Rules =====")
    for rule in sorted(rules, key=lambda x: -x["confidence"]):
        print(f"{rule['lhs']} -> {rule['rhs']} "
              f"(support={rule['support']:.2f}, confidence={rule['confidence']:.2f}, count={rule['count']})")

    # Summary
    total_itemsets = sum(len(v) for v in frequent_itemsets.values())
    print("\n=== Brute Force Summary ===")
    print(f"Total frequent itemsets: {total_itemsets}")
    print(f"Total association rules: {len(rules)}")
    print(f"Total execution time: {total_time:.6f} seconds")

    print("\nBrute Force execution complete!")

if __name__ == "__main__":
    main()