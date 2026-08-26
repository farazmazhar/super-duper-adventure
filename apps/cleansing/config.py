"""Cleansing-app paths.

Re-exports the shared pydantic-settings config (apps/common/config.py).
The app never reads os.environ directly — everything comes from Settings,
which layers process env over the repo-root `.env` file.
"""

from __future__ import annotations

from apps.common.config import DATA_DIR, DB_PATH, settings

__all__ = ["DATA_DIR", "DB_PATH", "settings"]

# The 5 raw inputs, exactly as named in the assignment.
RAW_FILES = {
    "customers": "customers.csv",
    "support_tickets": "support_tickets.csv",
    "product_usage": "product_usage.csv",
    "customer_feedback": "customer_feedback.csv",
    "subscription_events": "subscription_events.csv",
}
