"""
Supermarket Billing and Sales Analysis System
Dataset : Kaggle - Supermarket Sales (supermarket_sales - Sheet1.csv)
Run     : python supermarket_analysis.py
Install : pip install pandas numpy matplotlib seaborn
"""

import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
CSV_NAMES = ["supermarket_sales - Sheet1.csv", "supermarket_sales_-_Sheet1.csv"]
TAX_RATE = 0.05
LOW_LIMIT = 300      # bills below this are "Low"
HIGH_LIMIT = 800     # bills from this up are "High"
BRANCH_NAMES = {"A": "Yangon", "B": "Mandalay", "C": "Naypyitaw"}
CHART_FILES = []     # names of chart images saved by make_charts()


# ----------------------------------------------------------------------
# 1. Load the CSV (stop with a message if the file is missing)
# ----------------------------------------------------------------------
def load_data():
    for name in CSV_NAMES:
        if os.path.exists(name):
            return pd.read_csv(name)
    print("ERROR: CSV file not found. Keep 'supermarket_sales - Sheet1.csv'")
    print("in the same folder as this program.")
    sys.exit()


# ----------------------------------------------------------------------
# 2. Clean the data (done only once)
# ----------------------------------------------------------------------
def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()                      # trim names

    # remove duplicates and blank invoices
    df = df.drop_duplicates()
    df = df.dropna(subset=["Invoice ID"])

    # fix types, fill gaps with the median
    for col in ["Unit price", "Quantity", "Rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df["Quantity"] = df["Quantity"].astype(int)

    # fill missing text columns with the most common value
    for col in ["Branch", "City", "Customer type", "Gender",
                "Product line", "Payment"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # drop impossible rows
    bad = (df["Unit price"] <= 0) | (df["Quantity"] <= 0)
    df = df[~bad].copy()

    # fix dates and times
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Time"] = df["Time"].astype(str).str.strip()

    # recalculate the money columns
    df["cogs"] = (df["Unit price"] * df["Quantity"]).round(2)
    df["Tax 5%"] = df["cogs"] * TAX_RATE
    df["Total"] = df["cogs"] + df["Tax 5%"]
    df["gross income"] = df["Tax 5%"]

    # new columns: Hour, Month, Weekday, Bill size
    df["Hour"] = df["Time"].str.split(":").str[0].astype(int)
    df["Month"] = df["Date"].dt.month_name()
    df["Weekday"] = df["Date"].dt.day_name()
    df["Bill size"] = pd.cut(df["Total"],
                             bins=[0, LOW_LIMIT, HIGH_LIMIT, float("inf")],
                             labels=["Low", "Medium", "High"], right=False)

    # report outliers (IQR rule) but keep them
    q1, q3 = np.percentile(df["Total"], [25, 75])
    limit = q3 + 1.5 * (q3 - q1)
    outliers = int((df["Total"] > limit).sum())

    df = df.reset_index(drop=True)
    return df, limit, outliers


# ----------------------------------------------------------------------
# Classes: encapsulation, abstraction, inheritance, polymorphism
# ----------------------------------------------------------------------
class SalesData:                                # encapsulation
    def __init__(self, df):
        self.__df = df                          # private data

    @property
    def df(self):
        return self.__df.copy()


class Report(SalesData, ABC):                   # abstraction
    @abstractmethod
    def show(self):
        pass


class SalesReport(Report):                      # inheritance
    def show(self):                             # overriding
        d = self.df
        print("\n===== SALES SUMMARY =====")
        print(f"Total invoices    : {len(d)}")
        print(f"Total items sold  : {d['Quantity'].sum()}")
        print(f"Total sales       : {d['Total'].sum():,.2f}")
        print(f"Total before tax  : {d['cogs'].sum():,.2f}")
        print(f"Total tax/income  : {d['Tax 5%'].sum():,.2f}")
        print(f"Average bill      : {d['Total'].mean():,.2f}")
        print(f"Highest / lowest  : {d['Total'].max():,.2f} / "
              f"{d['Total'].min():,.2f}")
        print(f"Average rating    : {d['Rating'].mean():.2f}")
        sizes = d["Bill size"].value_counts()
        print("\nBills by size:")
        print("  " + "  ".join(f"{k} {sizes[k]}"
                               for k in ["Low", "Medium", "High"]))


class ProductReport(Report):                    # inheritance
    def show(self):                             # overriding
        d = self.df
        table = d.groupby("Product line").agg(
            Invoices=("Invoice ID", "count"),
            Qty=("Quantity", "sum"),
            Sales=("Total", "sum"))
        table["Share %"] = table["Sales"] / table["Sales"].sum() * 100
        table = table.sort_values("Invoices", ascending=False)

        print("\n===== PRODUCT LINES =====")
        print(f"{'Product line':<24}{'Invoices':>9}{'Qty':>7}"
              f"{'Sales':>13}{'Share %':>9}")
        for name, row in table.iterrows():
            print(f"{name:<24}{int(row['Invoices']):>9}{int(row['Qty']):>7}"
                  f"{row['Sales']:>13,.2f}{row['Share %']:>9.1f}")
        top = table.index[0]
        print(f"\nMost frequent: {top} ({int(table.loc[top, 'Invoices'])} "
              f"invoices)")
        best = table["Sales"].idxmax()
        print(f"Highest sales: {best} ({table.loc[best, 'Sales']:,.2f})")


# ----------------------------------------------------------------------
# Option 1: print a bill
# ----------------------------------------------------------------------
class Bill:
    def __init__(self, row):
        self.row = row

    def print_bill(self):
        r = self.row
        line = "-" * 34
        print("\n" + line)
        print(f"Invoice : {r['Invoice ID']}")
        print(f"Date    : {r['Date'].date()}  Time: {r['Time']}")
        print(f"Branch  : {r['Branch']} ({BRANCH_NAMES.get(r['Branch'], r['City'])})")
        print(f"Customer: {r['Customer type']}, {r['Gender']}")
        print(f"Payment : {r['Payment']}")
        print(line)
        print(r["Product line"])
        qty_price = f"  {r['Quantity']} x {r['Unit price']:.2f}"
        print(f"{qty_price:<22}{r['cogs']:>12,.2f}")
        print(line)
        print(f"{'Tax (5%)':<22}{r['Tax 5%']:>12,.2f}")
        print(f"{'TOTAL':<22}{r['Total']:>12,.2f}")
        print(f"Rating given: {r['Rating']}")
        print(line)


def bill_option(df):
    inv = input("Enter invoice number (e.g. 750-67-8428): ").strip()
    match = df[df["Invoice ID"] == inv]
    if match.empty:
        print("Invoice not found.")
    else:
        Bill(match.iloc[0]).print_bill()


# ----------------------------------------------------------------------
# Option 4: sales by group
# ----------------------------------------------------------------------
def group_report(df):
    print("\n===== SALES BY GROUP =====")
    for col in ["Branch", "Customer type", "Gender", "Payment",
                "Month", "Weekday"]:
        g = df.groupby(col)["Total"].agg(["count", "sum", "mean"])
        g = g.sort_values("sum", ascending=False)
        print(f"\nBy {col}:")
        print(f"  {'':<14}{'Invoices':>9}{'Sales':>13}{'Avg bill':>10}")
        for name, row in g.iterrows():
            print(f"  {str(name):<14}{int(row['count']):>9}"
                  f"{row['sum']:>13,.2f}{row['mean']:>10.2f}")


# ----------------------------------------------------------------------
# Option 5: tuples, sets, dictionaries, lambda, recursion
# ----------------------------------------------------------------------
def tuple_records(df):
    """Turn the first 5 invoices into tuples."""
    return [(r["Invoice ID"], r["Product line"], r["Quantity"],
             round(r["Total"], 2))
            for _, r in df.head(5).iterrows()]


def set_demo(df):
    """Sets keep only unique values and support set operations."""
    cities = set(df["City"])
    products = set(df["Product line"])
    ewallet = set(df[df["Payment"] == "Ewallet"]["Product line"])
    cash = set(df[df["Payment"] == "Cash"]["Product line"])
    return cities, products, ewallet & cash


def running_total(values):
    """Recursion: add up a list one item at a time."""
    if not values:
        return 0
    return values[0] + running_total(values[1:])


def demo_option(df):
    print("\n===== TUPLES, SETS, DICTIONARIES =====")

    print("\nTuples (invoice, product line, qty, total):")
    for t in tuple_records(df):
        print(" ", t)

    cities, products, both = set_demo(df)
    print("\nSet of cities           :", sorted(cities))
    print("Number of product lines :", len(products))
    print("Sold for both Ewallet and Cash:", len(both), "product lines")

    totals = {}                                  # dictionary
    for _, r in df.iterrows():
        totals[r["Branch"]] = totals.get(r["Branch"], 0) + r["Total"]
    print("\nDictionary of sales by branch:")
    for k, v in sorted(totals.items()):
        print(f"  {k} ({BRANCH_NAMES[k]}): {v:,.2f}")

    to_k = lambda x: round(x / 1000, 1)          # lambda
    print("\nBranch sales in thousands:", {k: to_k(v) for k, v in totals.items()})

    print("Recursive sum of first 10 bills:",
          f"{running_total(list(df['Total'].head(10))):,.2f}")

    # type casting and operators
    qty_text = str(int(df["Quantity"].sum()))
    print("Type casting: items sold as text =", repr(qty_text),
          "-> as int + 1 =", int(qty_text) + 1)


# ----------------------------------------------------------------------
# Option 6: six charts
# ----------------------------------------------------------------------
def make_charts(df):
    sns.set_theme(style="whitegrid")
    CHART_FILES.clear()

    def finish(name):
        plt.tight_layout()
        plt.savefig(name, dpi=120)
        CHART_FILES.append(name)
        plt.show()

    # 1. Total sales by product line
    plt.figure(figsize=(8, 5))
    s = df.groupby("Product line")["Total"].sum().sort_values(ascending=False)
    sns.barplot(x=s.index, y=s.values, color="#2E75B6")
    plt.title("1. Total sales by product line")
    plt.xlabel("Product line")
    plt.ylabel("Total sales")
    plt.xticks(rotation=45, ha="right")
    finish("chart1_sales_by_product.png")

    # 2. Invoices by product line
    plt.figure(figsize=(8, 5))
    c = df["Product line"].value_counts()
    sns.barplot(x=c.index, y=c.values, color="#70AD47")
    plt.title("2. Invoices by product line")
    plt.xlabel("Product line")
    plt.ylabel("Number of invoices")
    plt.xticks(rotation=45, ha="right")
    finish("chart2_invoices_by_product.png")

    # 3. Daily sales trend
    plt.figure(figsize=(9, 5))
    daily = df.groupby("Date")["Total"].sum()
    plt.plot(daily.index, daily.values, color="#C55A11")
    plt.title("3. Daily sales trend")
    plt.xlabel("Date")
    plt.ylabel("Total sales")
    plt.xticks(rotation=45, ha="right")
    finish("chart3_daily_trend.png")

    # 4. Payment methods
    plt.figure(figsize=(6, 6))
    p = df["Payment"].value_counts()
    plt.pie(p.values, labels=p.index, autopct="%1.1f%%", startangle=90)
    plt.title("4. Payment methods")
    finish("chart4_payment.png")

    # 5. Sales by branch and customer type
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="Branch", y="Total", hue="Customer type",
                estimator=sum, errorbar=None)
    plt.title("5. Sales by branch and customer type")
    plt.xlabel("Branch")
    plt.ylabel("Total sales")
    finish("chart5_branch_customer.png")

    # 6. Sales by hour of the day
    plt.figure(figsize=(8, 5))
    h = df.groupby("Hour")["Total"].sum()
    plt.bar(h.index, h.values, color="#7030A0")
    plt.title("6. Sales by hour of the day")
    plt.xlabel("Hour of the day")
    plt.ylabel("Total sales")
    plt.xticks(h.index)
    finish("chart6_hour.png")

    print("\n6 charts shown and saved as PNG files in this folder.")


# ----------------------------------------------------------------------
# Option 7: save files
# ----------------------------------------------------------------------
def save_summary(df):
    df.to_csv("cleaned_sales.csv", index=False)

    d = df
    with open("sales_summary.txt", "w", encoding="utf-8") as f:
        f.write("SUPERMARKET SALES SUMMARY\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total invoices   : {len(d)}\n")
        f.write(f"Total items sold : {d['Quantity'].sum()}\n")
        f.write(f"Total sales      : {d['Total'].sum():,.2f}\n")
        f.write(f"Average bill     : {d['Total'].mean():,.2f}\n")
        f.write(f"Average rating   : {d['Rating'].mean():.2f}\n\n")
        f.write("Sales by product line:\n")
        f.write(d.groupby("Product line")["Total"].sum()
                .sort_values(ascending=False).round(2).to_string())
        f.write("\n\nSales by branch:\n")
        f.write(d.groupby("Branch")["Total"].sum().round(2).to_string())
        f.write("\n")

    print("Saved: cleaned_sales.csv and sales_summary.txt")


# ----------------------------------------------------------------------
# Main program: menu loop
# ----------------------------------------------------------------------
def main():
    raw = load_data()
    df, limit, outliers = clean_data(raw)

    print("Supermarket Billing and Sales Analysis System")
    print(f"Loaded {len(raw)} rows, kept {len(df)} after cleaning "
          f"({raw.shape[1]} -> {df.shape[1]} columns).")
    print(f"Outlier bills (above {limit:,.2f}): {outliers} (kept)")

    reports = [SalesReport(df), ProductReport(df)]   # polymorphism

    while True:
        print("\n========== MENU ==========")
        print("1. Print bill")
        print("2. Sales summary")
        print("3. Frequent products")
        print("4. Sales by group")
        print("5. Tuples, sets, dictionaries")
        print("6. Charts")
        print("7. Save files")
        print("8. Exit")

        try:
            choice = int(input("Enter choice (1-8): "))
        except ValueError:
            print("Invalid choice. Please enter a number from 1 to 8.")
            continue

        if choice == 1:
            bill_option(df)
        elif choice == 2:
            reports[0].show()
        elif choice == 3:
            reports[1].show()
        elif choice == 4:
            group_report(df)
        elif choice == 5:
            demo_option(df)
        elif choice == 6:
            make_charts(df)
        elif choice == 7:
            save_summary(df)
        elif choice == 8:
            print("Thank you. Goodbye!")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
