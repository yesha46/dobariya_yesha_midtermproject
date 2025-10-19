import pandas as pd
import os
import time
import warnings
from mlxtend.frequent_patterns import apriori, association_rules
warnings.filterwarnings("ignore")

# Get available CSV files
def get_csv_files(folder_path):
    return [f for f in os.listdir(folder_path)
            if f.endswith(".csv") and not f.startswith("frequent_itemsets_") and not f.startswith("rules_")]

# Load Transactions
def load_transactions(file_path):
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'items' not in df.columns:
        raise ValueError("CSV must contain a column named 'Items'")
    df['Items'] = df['items'].apply(lambda x: [i.strip() for i in x.split(',')])
    all_items = sorted(set(item for sublist in df['Items'] for item in sublist))
    one_hot = pd.DataFrame([{item: (item in trans) for item in all_items} for trans in df['Items']])
    return one_hot

# Get valid float input between 0 and 1
def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if 0 < value <= 1:
                return value
            else:
                print("Please enter a number between 0 and 1 (exclusive of 0).")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Main
if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), "data") 
    csv_files = get_csv_files(folder_path)

    if not csv_files:
        print("No CSV files found in the data folder.")
        exit()

    print("Available datasets:")
    for i, f in enumerate(csv_files, 1):
        print(f"{i}. {f}")
    
    try:
        choice = int(input("\nSelect a dataset by number: "))
        if choice < 1 or choice > len(csv_files):
            raise ValueError
        dataset = csv_files[choice - 1]
        file_path = os.path.join(folder_path, dataset)
        print(f"\nSelected dataset: {dataset}")
    except ValueError:
        print("Invalid input. Please enter a valid dataset number.")
        exit()

    # Input validation
    min_support = get_float_input("Enter minimum support (e.g., 0.2 for 20%): ")
    min_conf = get_float_input("Enter minimum confidence (e.g., 0.6 for 60%): ")

    print(f"\nRunning Apriori Algorithm on '{dataset}' ...")

    # Total execution start
    start_total = time.perf_counter()

    # Frequent itemset mining
    df = load_transactions(file_path)
    total_transactions = len(df)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    frequent_itemsets['count'] = (frequent_itemsets['support'] * total_transactions).astype(int)

    # Association rule generation
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    rules['count'] = (rules['support'] * total_transactions).astype(int)

    # Total execution end
    end_total = time.perf_counter()
    total_time = end_total - start_total

    # Print frequent itemsets
    for k in sorted(frequent_itemsets['itemsets'].apply(len).unique()):
        print(f"\n===== {k}-Frequent Itemsets =====")
        subset = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == k]
        for _, row in subset.iterrows():
            print(f"{tuple(row['itemsets'])}: support={row['support']:.2f}, count={row['count']}")

    # Print association rules
    print("\n===== Association Rules =====")
    for _, r in rules.iterrows():
        print(f"{tuple(r['antecedents'])} -> {tuple(r['consequents'])} "
              f"(support={r['support']:.2f}, confidence={r['confidence']:.2f}, count={r['count']})")

    # Summary
    total_itemsets = len(frequent_itemsets)
    total_rules = len(rules)
    print("\n=== Apriori Summary ===")
    print(f"Total frequent itemsets: {total_itemsets}")
    print(f"Total association rules: {total_rules}")
    print(f"Total execution time: {total_time:.6f} seconds")

    print("\nApriori execution complete!")