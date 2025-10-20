# dobariya_yesha_midtermproject
**Name:** Yesha Dobariya  
**Email:** yd326@njit.edu  
**Course:** CS634 - Data Mining  
**Instructor:** Dr. Yasser Abduallah  

## Project Overview
This project focuses on **Frequent Itemset Mining** and **Association Rule Learning** core concepts in data mining used to discover meaningful relationships among items in transactional datasets. The goal is to explore how different algorithms identify frequent itemsets and generate association rules, revealing patterns of items often purchased together. Three major algorithms were implemented and compared for performance, efficiency, and scalability:

### 1. Brute Force
- Generates all possible item combinations.
- Counts occurrences to identify frequent itemsets based on support threshold.
- Serves as a conceptual baseline but is computationally expensive for larger datasets.

### 2. Apriori
- Utilizes the **Apriori Property**: “If an itemset is frequent, all its subsets must also be frequent.”
- Iteratively prunes infrequent subsets to improve efficiency.
- Performs well on moderate-sized datasets but requires multiple scans.

### 3. FP-Growth
- Builds a **Frequent Pattern Tree (FP-Tree)** to compress transactions.
- Mines frequent itemsets directly without candidate generation.
- Fastest and most memory-efficient for large or complex datasets.

Each algorithm was applied to five manually created transactional datasets representing different store types **Amazon**, **BestBuy**, **K-Mart**, **Nike**, and a **Generic** dataset simulating real-world shopping behavior. The comparison highlights the trade-offs between simplicity (Brute Force), interpretability (Apriori), and scalability (FP-Growth).

## Datasets
Each dataset contains at least **five unique items** and **20 deterministic transactions**, designed to mimic realistic shopping behaviors.

| Store     | Focus Area                        | Example Items                                                                 |
|------------|-----------------------------------|--------------------------------------------------------------------------------|
| Amazon     | Programming & Books               | Java For Dummies, Head First Java, Android Programming                         |
| BestBuy    | Electronics & Accessories         | Laptop, Printer, Flash Drive, Microsoft Office, Antivirus                      |
| K-Mart     | Home & Bedding Products           | Quilts, Bedspreads, Sheets, Decorative Pillows                                 |
| Nike       | Sportswear & Apparel              | Running Shoes, Sweatshirts, Tech Pants, Hoodies                                |
| Generic    | Abstract Items for Scalability    | A, B, C, D, E, F                                                              |

All datasets are saved as CSV files and stored in the `data/` folder.

## Environment & Installation
### 1. Recommended Versions & Prerequisites
- **Operating System:** Windows / macOS / Linux  
- **Python Version:** Python 3.8 or higher    
- **Shell:** PowerShell / bash / cmd  
- **Tools Required:** Jupyter Notebook
 
### 2. Create a Virtual Environment
- It’s best practice to create a virtual environment to isolate dependencies for this project.
- Windows: Run **py -3 -m venv .venv** and then **.\\.venv\\Scripts\\activate**
- macOS/Linux: Run **python3 -m venv .venv** and then **source .venv/bin/activate**

 ### 3. Install Required Libraries
- Open a terminal or command prompt and install the library by running: **pip install pandas mlxtend**
- You can install all the necessary dependencies using the provided `requirements.txt` file using the command: **pip install -r requirements.txt**.

## How to Run the Code
### Option 1: Run Python Scripts
- Run Brute Force algorithm: **python src/brute_force.py**
- Run Apriori algorithm: **python src/apriori_runner.py**
- Run FP-Growth algorithm: **python src/fpgrowth_runner.py**

### Option 2: Run via Jupyter Notebook
- jupyter notebook: **notebooks/project_demo.ipynb**

## Conclusion
This project demonstrates how data mining techniques specifically **Brute Force**, **Apriori**, and **FP-Growth** can uncover meaningful associations within transactional data.  
Through experimentation across multiple store-based datasets, it was observed that:

- **FP-Growth** achieved the best performance and scalability due to its tree-based structure.  
- **Apriori** offered a good balance between interpretability and efficiency but required multiple dataset scans.  
- **Brute Force** served as a valuable baseline for understanding the core logic behind frequent itemset mining.  

Overall, the project highlights how algorithmic optimization significantly impacts computational efficiency.
