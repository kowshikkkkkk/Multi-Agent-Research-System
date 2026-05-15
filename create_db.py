import sqlite3
from pathlib import Path

db_path = Path("financial_data.db")

# Remove existing DB if it exists (clean slate)
if db_path.exists():
    db_path.unlink()

conn = sqlite3.connect("financial_data.db")
cursor = conn.cursor()

# Create quarterly financial data table
cursor.execute('''
    CREATE TABLE quarterly_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT NOT NULL,
        quarter TEXT NOT NULL,
        year INTEGER NOT NULL,
        revenue_usd_millions REAL,
        operating_profit_usd_millions REAL,
        net_profit_usd_millions REAL,
        employee_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# Infosys quarterly data (2024 YTD)
quarterly_records = [
    ("Infosys", "Q1", 2024, 2100.5, 420.3, 380.5, 319000),
    ("Infosys", "Q2", 2024, 2250.2, 480.1, 425.8, 320500),
    ("Infosys", "Q3", 2024, 2180.8, 450.6, 410.2, 321200),
    ("Infosys", "Q4", 2023, 2050.3, 390.5, 355.1, 318000),
]

cursor.executemany(
    '''INSERT INTO quarterly_data 
       (company, quarter, year, revenue_usd_millions, operating_profit_usd_millions, 
        net_profit_usd_millions, employee_count) 
       VALUES (?, ?, ?, ?, ?, ?, ?)''',
    quarterly_records
)

# Create segment revenue table (business lines)
cursor.execute('''
    CREATE TABLE segment_revenue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT NOT NULL,
        segment_name TEXT NOT NULL,
        year INTEGER NOT NULL,
        revenue_usd_millions REAL,
        growth_percent REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

segment_records = [
    ("Infosys", "Financial Services", 2024, 850.5, 8.2),
    ("Infosys", "Retail & Consumer", 2024, 620.3, 5.1),
    ("Infosys", "Manufacturing", 2024, 480.2, 12.5),
    ("Infosys", "Healthcare & Life Sciences", 2024, 360.8, 15.3),
    ("Infosys", "Telecom & Media", 2024, 240.5, 3.2),
    ("Infosys", "Energy & Utilities", 2024, 180.4, -2.1),
]

cursor.executemany(
    '''INSERT INTO segment_revenue 
       (company, segment_name, year, revenue_usd_millions, growth_percent) 
       VALUES (?, ?, ?, ?, ?)''',
    segment_records
)

# Create key metrics table
cursor.execute('''
    CREATE TABLE key_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        year INTEGER NOT NULL,
        value REAL,
        unit TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

metrics_records = [
    ("Infosys", "Operating Margin", 2024, 21.2, "%"),
    ("Infosys", "Net Margin", 2024, 18.5, "%"),
    ("Infosys", "Return on Equity", 2024, 32.4, "%"),
    ("Infosys", "Debt-to-Equity", 2024, 0.15, "ratio"),
    ("Infosys", "Attrition Rate", 2024, 15.8, "%"),
]

cursor.executemany(
    '''INSERT INTO key_metrics 
       (company, metric_name, year, value, unit) 
       VALUES (?, ?, ?, ?, ?)''',
    metrics_records
)

conn.commit()

# Verify data insertion
print("✓ Database created successfully\n")
print("Tables created:")
print("  1. quarterly_data — Q1-Q4 financial snapshots")
print("  2. segment_revenue — Business segment breakdown")
print("  3. key_metrics — Profitability, margin, attrition KPIs\n")

# Show sample records
print("Sample quarterly data:")
result = cursor.execute("SELECT company, quarter, year, revenue_usd_millions, employee_count FROM quarterly_data LIMIT 3")
for row in result:
    print(f"  {row[0]} {row[1]} {row[2]}: ${row[3]:.1f}M revenue, {row[4]:,} employees")

print("\nSample segment data:")
result = cursor.execute("SELECT company, segment_name, revenue_usd_millions, growth_percent FROM segment_revenue LIMIT 3")
for row in result:
    print(f"  {row[1]}: ${row[2]:.1f}M ({row[3]:+.1f}% growth)")

conn.close()
print("\n✓ financial_data.db is ready to use")