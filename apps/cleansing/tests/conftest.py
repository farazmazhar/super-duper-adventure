"""Shared fixture: tiny synthetic CSVs covering every clean rule.

Rows are chosen to exercise:
- customers: dup (keep last), case-normalize status, NULL revenue kept
- support_tickets: exact dup dropped, case-normalize category/priority,
  negative resolution -> NULL, NULL resolution/satisfaction kept
- customer_feedback: rating >5 -> NULL, NULL text kept
- product_usage: NULL avg_session_duration kept, trend windows
- subscription_events: pass-through
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
APP_DIR = Path(__file__).resolve().parents[1]  # apps/cleansing

CUSTOMERS = """customer_id,customer_name,customer_segment,country,subscription_plan,monthly_revenue,account_created_at,account_status
CUST-0001,Alpha,Mid-Market,France,Business,1000.00,2023-01-01,ACTIVE
CUST-0001,Alpha,Mid-Market,France,Business,1100.00,2023-01-01,active
CUST-0002,Beta,SMB,Germany,Team,,2024-02-02,CANCELED
CUST-0003,Gamma,Enterprise,USA,Enterprise,2500.00,2024-03-03,paused
"""

TICKETS = """ticket_id,customer_id,created_at,subject,message,category,priority,resolution_time_hours,status,satisfaction_score
TCK-0001,CUST-0001,2026-05-01 10:00,Bug,It crashes.,Bug,High,5.0,resolved,4.0
TCK-0001,CUST-0001,2026-05-01 10:00,Bug,It crashes.,Bug,High,5.0,resolved,4.0
TCK-0002,CUST-0002,2026-05-02 11:00,Question,,General_question,low,,open,
TCK-0003,CUST-0003,2026-05-03 12:00,Neg time,Weird.,billing,medium,-5.0,resolved,3.0
"""

FEEDBACK = """feedback_id,customer_id,created_at,feedback_text,feedback_source,rating
FDB-0001,CUST-0001,2026-05-10 09:00,Love the search feature,support_chat,5
FDB-0002,CUST-0002,2026-05-11 09:00,,email,8
FDB-0003,CUST-0003,2026-05-12 09:00,Okay,app_store_review,3
"""

USAGE = """customer_id,date,active_users,sessions,feature_usage,errors,average_session_duration
CUST-0001,2026-05-25,10,100,search,1,8.0
CUST-0001,2026-06-20,12,150,dashboards,0,9.0
CUST-0001,2026-06-25,15,200,reports,2,
CUST-0002,2026-05-25,5,50,api,0,4.0
"""

SUB_EVENTS = """customer_id,event_date,event_type,previous_plan,new_plan,revenue_change
CUST-0001,2026-07-01,upgrade,Business,Enterprise,500.0
CUST-0002,2026-07-02,cancellation,Team,Free,-200.0
"""


@pytest.fixture(scope="session")
def con(tmp_path_factory) -> duckdb.DuckDBPyConnection:
    """Build a clean DuckDB in a temp dir and run the real pipeline against fixtures."""
    data_dir = tmp_path_factory.mktemp("data")
    for name, content in [
        ("customers.csv", CUSTOMERS),
        ("support_tickets.csv", TICKETS),
        ("product_usage.csv", USAGE),
        ("customer_feedback.csv", FEEDBACK),
        ("subscription_events.csv", SUB_EVENTS),
    ]:
        (data_dir / name).write_text(content)

    db_path = tmp_path_factory.mktemp("db") / "intelligence.duckdb"
    con = duckdb.connect(str(db_path))
    sql = (APP_DIR / "pipeline.sql").read_text().replace("{{data_dir}}", str(data_dir))
    con.execute(sql)
    yield con
    con.close()
