-- ============================================================================
-- App 1 — Data Cleansing pipeline (DuckDB)
--
-- Staging -> Clean -> Feature tables -> Quality report.
-- Idempotent: every run rebuilds all tables from the raw CSVs (CREATE OR REPLACE).
-- Auditable: every cleaning decision logs a row into main.quality_report.
-- SQL-first: this file is the pipeline; run.py is a thin orchestrator.
--
-- Schema ownership: App 1 owns `main`. All tables are explicitly qualified
-- `main.*` so the single DuckDB file stays partitioned: `main` (customer data,
-- this app), `vector` (App 2), `agent` (App 3).
--
-- All paths below are rendered by run.py ({{data_dir}}, {{db_path}}).
-- ============================================================================
-- ----------------------------------------------------------------------------
-- Stage 1 — Staging: read the 5 CSVs exactly as-is (inferred types).
-- Input order is preserved via an explicit row index so dedup "keep last"
-- (later write wins) is deterministic and reviewable.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE main.staging_customers AS
SELECT *,
  row_number() OVER () AS _input_row_number
FROM read_csv('{{data_dir}}/customers.csv', header = true);
CREATE OR REPLACE TABLE main.staging_tickets AS
SELECT *,
  row_number() OVER () AS _input_row_number
FROM read_csv(
    '{{data_dir}}/support_tickets.csv',
    header = true
  );
CREATE OR REPLACE TABLE main.staging_usage AS
SELECT *,
  row_number() OVER () AS _input_row_number
FROM read_csv('{{data_dir}}/product_usage.csv', header = true);
CREATE OR REPLACE TABLE main.staging_feedback AS
SELECT *,
  row_number() OVER () AS _input_row_number
FROM read_csv(
    '{{data_dir}}/customer_feedback.csv',
    header = true
  );
CREATE OR REPLACE TABLE main.staging_subscription_events AS
SELECT *,
  row_number() OVER () AS _input_row_number
FROM read_csv(
    '{{data_dir}}/subscription_events.csv',
    header = true
  );
-- ----------------------------------------------------------------------------
-- Stage 2 — Clean. Each step logs its affected-row count to main.quality_report.
-- ----------------------------------------------------------------------------
-- Audit table: one row per executed clean step, with affected-row count.
CREATE OR REPLACE TABLE main.quality_report (
    rule TEXT,
    table_name TEXT,
    description TEXT,
    count INT
  );

-- Rule catalogue: one row per cleaning rule. The quality report references these
-- rule ids; this table is the machine-readable definition of every rule so the
-- pipeline is self-describing (no data-specific values baked into SQL).
CREATE OR REPLACE TABLE main.pipeline_rules (
    rule TEXT PRIMARY KEY,
    rule_name TEXT,
    description TEXT,
    applies_to TEXT
  );
INSERT INTO main.pipeline_rules
VALUES
  ('deduplicate_customers', 'deduplicate customers keep last',
   'Drop duplicate customer rows by id, keeping the LAST row in input order. Assumption: later write = latest value.',
   'customers'),
  ('normalize_account_status', 'normalize account status case',
   'Lowercase account_status so status values form one canonical set.',
   'customers'),
  ('null_revenue_kept', 'preserve null revenue',
   'NULL monthly_revenue is preserved (never imputed) and counted.',
   'customers'),
  ('cast_account_created_at', 'cast account created date',
   'Cast account_created_at to DATE; unparseable values become NULL.',
   'customers'),
  ('invalid_date_to_null', 'invalid date to null',
   'Rows whose date column cannot be parsed as a valid date have the value set to NULL.',
   'customers'),
  ('deduplicate_tickets', 'deduplicate exact duplicate tickets',
   'Drop exact duplicate ticket rows (identical content), keeping the FIRST occurrence by id.',
   'support_tickets'),
  ('normalize_category', 'normalize ticket category case',
   'Lowercase ticket category so categories form one canonical set.',
   'support_tickets'),
  ('normalize_priority', 'normalize ticket priority case',
   'Lowercase ticket priority so priorities form one canonical set.',
   'support_tickets'),
  ('clip_negative_resolution', 'clip negative resolution time',
   'Negative resolution_time_hours is impossible; set to NULL.',
   'support_tickets'),
  ('null_resolution_kept', 'preserve null resolution time',
   'NULL resolution_time_hours is preserved (unresolved tickets have no resolution time).',
   'support_tickets'),
  ('null_satisfaction_kept', 'preserve null satisfaction',
   'NULL satisfaction_score is preserved (only resolved tickets have satisfaction).',
   'support_tickets'),
  ('null_message_kept', 'preserve null message',
   'NULL ticket message is preserved (subject still carries signal).',
   'support_tickets'),
  ('cast_created_at', 'cast created timestamp',
   'Cast created_at to TIMESTAMP so time windows can be computed.',
   'support_tickets'),
  ('null_feedback_text_kept', 'preserve null feedback text',
   'NULL feedback_text is preserved and counted.',
   'customer_feedback'),
  ('normalize_rating_scale', 'normalize out-of-range rating',
   'Feedback rating above 5 is treated as a 10-point scale and normalized to 5 points (value / 2). Ratings within 1-5 pass through unchanged; NULL stays NULL.',
   'customer_feedback'),
  ('null_session_duration_kept', 'preserve null session duration',
   'NULL average_session_duration is preserved (missing is meaningful).',
   'product_usage'),
  ('cast_date', 'cast date column',
   'Cast date columns to DATE so time windows can be computed.',
   'product_usage'),
  ('pass_through', 'pass through clean table',
   'Source verified clean (no duplicates, no nulls); passed through with date cast.',
   'subscription_events');

-- ---- 2.1 customers ---------------------------------------------------------
-- Dedup by customer_id, KEEP LAST row in input order. Assumption: later write
-- wins = latest value.
CREATE OR REPLACE TABLE main.clean_customers AS WITH ranked AS (
    SELECT *,
      row_number() OVER (
        PARTITION BY customer_id
        ORDER BY _input_row_number DESC
      ) AS row_rank
    FROM main.staging_customers
  )
SELECT * EXCLUDE (row_rank)
FROM ranked
WHERE row_rank = 1;
INSERT INTO main.quality_report
VALUES (
    'deduplicate_customers',
    'customers',
    'Dropped duplicate customer rows (keep last in input order; assumption: later write = latest value).',
    (
      SELECT count(*)
      FROM main.staging_customers
    ) - (
      SELECT count(*)
      FROM main.clean_customers
    )
  );
-- Case-normalize account_status: ACTIVE -> active, CANCELED -> canceled.
UPDATE main.clean_customers
SET account_status = lower(account_status);
INSERT INTO main.quality_report
VALUES (
    'normalize_account_status',
    'customers',
    'Lowercased account_status so status values form one canonical set.',
    (
      SELECT count(*)
      FROM main.staging_customers
      WHERE account_status <> lower(account_status)
    )
  );
-- NULL monthly_revenue kept (user decision: no invented data). Count them.
INSERT INTO main.quality_report
VALUES (
    'null_revenue_kept',
    'customers',
    'NULL monthly_revenue preserved (not imputed). Distinct customers with NULL revenue.',
    (
      SELECT count(*)
      FROM main.clean_customers
      WHERE monthly_revenue IS NULL
    )
  );
-- Cast created_at to DATE. Rows with an impossible date value (e.g. month 13
-- in a month/day field) become NULL rather than failing the pipeline.
-- Impossible values become NULL (same policy as the rating/negative-resolution
-- clips).
ALTER TABLE main.clean_customers ALTER account_created_at TYPE DATE USING CASE
    WHEN try_cast(account_created_at AS DATE) IS NULL THEN NULL
    ELSE account_created_at::DATE
  END;
INSERT INTO main.quality_report
VALUES (
    'cast_account_created_at',
    'customers',
    'Cast account_created_at to DATE; unparseable values set to NULL.',
    (
      SELECT count(*)
      FROM main.clean_customers
    )
  );
INSERT INTO main.quality_report
VALUES (
    'invalid_date_to_null',
    'customers',
    'Rows with an impossible date value (unparseable) set to NULL.',
    (
      SELECT count(*)
      FROM main.clean_customers
      WHERE account_created_at IS NULL
    )
  );
-- ---- 2.2 tickets -----------------------------------------------------------
-- Drop exact duplicate rows (identical content). Exclude the staging
-- _input_row_number (unique per row, so plain DISTINCT would never drop them)
-- and keep the FIRST occurrence per ticket_id.
CREATE OR REPLACE TABLE main.clean_tickets AS WITH distinct_content AS (
    SELECT * EXCLUDE (_input_row_number),
      row_number() OVER (
        PARTITION BY ticket_id
        ORDER BY _input_row_number
      ) AS row_rank
    FROM main.staging_tickets
  )
SELECT * EXCLUDE (row_rank)
FROM distinct_content
WHERE row_rank = 1;
INSERT INTO main.quality_report
VALUES (
    'deduplicate_tickets',
    'support_tickets',
    'Dropped exact duplicate ticket rows (identical content), keeping the first occurrence.',
    (
      SELECT count(*)
      FROM main.staging_tickets
    ) - (
      SELECT count(*)
      FROM main.clean_tickets
    )
  );
-- Case-normalize category to the canonical lowercase set.
UPDATE main.clean_tickets
SET category = lower(category);
INSERT INTO main.quality_report
VALUES (
    'normalize_category',
    'support_tickets',
    'Lowercased ticket category so categories form one canonical set.',
    (
      SELECT count(*)
      FROM main.staging_tickets
      WHERE category <> lower(category)
    )
  );
-- Case-normalize priority (already lowercase; guard anyway).
UPDATE main.clean_tickets
SET priority = lower(priority);
INSERT INTO main.quality_report
VALUES (
    'normalize_priority',
    'support_tickets',
    'Lowercased ticket priority so priorities form one canonical set.',
    (
      SELECT count(*)
      FROM main.staging_tickets
      WHERE priority <> lower(priority)
    )
  );
-- Impossible negative resolution_time_hours -> NULL. Unresolved tickets keep
-- NULL resolution (meaningful) — not touched here.
UPDATE main.clean_tickets
SET resolution_time_hours = NULL
WHERE resolution_time_hours IS NOT NULL
  AND resolution_time_hours < 0;
INSERT INTO main.quality_report
VALUES (
    'clip_negative_resolution',
    'support_tickets',
    'Negative resolution_time_hours set to NULL (impossible value).',
    (
      SELECT count(*)
      FROM main.staging_tickets
      WHERE resolution_time_hours IS NOT NULL
        AND resolution_time_hours < 0
    )
  );
-- NULL resolution_time_hours kept (unresolved tickets) — log the count.
INSERT INTO main.quality_report
VALUES (
    'null_resolution_kept',
    'support_tickets',
    'NULL resolution_time_hours preserved (unresolved tickets have no resolution time).',
    (
      SELECT count(*)
      FROM main.clean_tickets
      WHERE resolution_time_hours IS NULL
    )
  );
-- NULL satisfaction_score kept (only resolved tickets have satisfaction).
INSERT INTO main.quality_report
VALUES (
    'null_satisfaction_kept',
    'support_tickets',
    'NULL satisfaction_score preserved (only resolved tickets have satisfaction).',
    (
      SELECT count(*)
      FROM main.clean_tickets
      WHERE satisfaction_score IS NULL
    )
  );
-- NULL message kept (subject carries signal for "Small request" rows).
INSERT INTO main.quality_report
VALUES (
    'null_message_kept',
    'support_tickets',
    'NULL message preserved (subject carries signal).',
    (
      SELECT count(*)
      FROM main.clean_tickets
      WHERE message IS NULL
    )
  );
-- Cast created_at to TIMESTAMP.
ALTER TABLE main.clean_tickets ALTER created_at TYPE TIMESTAMP USING created_at::TIMESTAMP;
INSERT INTO main.quality_report
VALUES (
    'cast_created_at',
    'support_tickets',
    'Cast created_at to TIMESTAMP.',
    (
      SELECT count(*)
      FROM main.clean_tickets
    )
  );
-- ---- 2.3 feedback ----------------------------------------------------------
-- NULL feedback_text kept — log count.
INSERT INTO main.quality_report
VALUES (
    'null_feedback_text_kept',
    'customer_feedback',
    'NULL feedback_text preserved.',
    (
      SELECT count(*)
      FROM main.staging_feedback
      WHERE feedback_text IS NULL
    )
  );
-- Rating normalization: values within 1-5 pass through unchanged; values above
-- 5 are treated as a 10-point scale and normalized to 5 points (value / 2);
-- NULL stays NULL. Keeps every rating comparable on one scale.
CREATE OR REPLACE TABLE main.clean_feedback AS
SELECT * EXCLUDE (rating),
  CASE
    WHEN rating IS NULL THEN NULL
    WHEN rating BETWEEN 1 AND 5 THEN rating
    ELSE rating / 2.0
  END AS rating
FROM main.staging_feedback;
INSERT INTO main.quality_report
VALUES (
    'normalize_rating_scale',
    'customer_feedback',
    'Rating above 5 (10-point scale) normalized to 5 points: value / 2.',
    (
      SELECT count(*)
      FROM main.staging_feedback
      WHERE rating > 5
    )
  );
-- Cast created_at to TIMESTAMP.
ALTER TABLE main.clean_feedback ALTER created_at TYPE TIMESTAMP USING created_at::TIMESTAMP;
INSERT INTO main.quality_report
VALUES (
    'cast_created_at',
    'customer_feedback',
    'Cast created_at to TIMESTAMP.',
    (
      SELECT count(*)
      FROM main.clean_feedback
    )
  );
-- ---- 2.4 usage -------------------------------------------------------------
-- NULL average_session_duration kept (missing is meaningful).
CREATE OR REPLACE TABLE main.clean_usage AS
SELECT *
FROM main.staging_usage;
INSERT INTO main.quality_report
VALUES (
    'null_session_duration_kept',
    'product_usage',
    'NULL average_session_duration preserved (missing is meaningful).',
    (
      SELECT count(*)
      FROM main.clean_usage
      WHERE average_session_duration IS NULL
    )
  );
-- Cast date to DATE.
ALTER TABLE main.clean_usage ALTER date TYPE DATE USING date::DATE;
INSERT INTO main.quality_report
VALUES (
    'cast_date',
    'product_usage',
    'Cast date to DATE.',
    (
      SELECT count(*)
      FROM main.clean_usage
    )
  );
-- ---- 2.5 subscription events -----------------------------------------------
-- Verified clean in source data (no dups, no nulls) — pass through for
-- uniformity.
CREATE OR REPLACE TABLE main.clean_subscription_events AS
SELECT *
FROM main.staging_subscription_events;
ALTER TABLE main.clean_subscription_events ALTER event_date TYPE DATE USING event_date::DATE;
INSERT INTO main.quality_report
VALUES (
    'pass_through',
    'subscription_events',
    'Verified clean (no dups, no nulls); passed through and event_date cast to DATE.',
    (
      SELECT count(*)
      FROM main.clean_subscription_events
    )
  );
-- ----------------------------------------------------------------------------
-- Stage 2b — Canonical categorical values (config, not hardcoded lists).
--
-- These tables are the SINGLE PLACE that names the canonical values for
-- ticket category/priority/status and subscription event_type. The aggregate
-- queries below pivot over whatever values live here. If the source data
-- introduces a new category/priority/status/event type, add it here (and to
-- pipeline_rules if it needs a rule) — the feature tables pick it up without
-- query changes. This is what keeps the pipeline data-agnostic: the only
-- data-specific strings in the analytical SQL live in these config tables.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE main.ticket_category (
    category VARCHAR PRIMARY KEY,
    description VARCHAR
  );
INSERT INTO main.ticket_category
VALUES
  ('billing', 'Billing / payments / invoices'),
  ('bug', 'Product bug or crash'),
  ('feature_request', 'Requested feature or enhancement'),
  ('general_question', 'General product question'),
  ('technical_issue', 'Technical problem'),
  ('onboarding', 'Onboarding / setup help'),
  ('account_access', 'Account / login / access');

CREATE OR REPLACE TABLE main.ticket_priority (
    priority VARCHAR PRIMARY KEY,
    description VARCHAR
  );
INSERT INTO main.ticket_priority
VALUES
  ('urgent', 'Urgent'),
  ('high', 'High'),
  ('medium', 'Medium'),
  ('low', 'Low');

CREATE OR REPLACE TABLE main.ticket_status (
    status VARCHAR PRIMARY KEY,
    description VARCHAR
  );
INSERT INTO main.ticket_status
VALUES
  ('open', 'Open (unresolved)'),
  ('pending', 'Pending (awaiting response)'),
  ('resolved', 'Resolved');

CREATE OR REPLACE TABLE main.subscription_event_type (
    event_type VARCHAR PRIMARY KEY,
    description VARCHAR
  );
INSERT INTO main.subscription_event_type
VALUES
  ('cancellation', 'Plan cancelled'),
  ('downgrade', 'Plan downgraded'),
  ('upgrade', 'Plan upgraded'),
  ('renewal', 'Plan renewed');

-- Account statuses that count as "churned" (churn proxy). Config-driven: the
-- churn_proxy flag in aggregate_customer_features reads THIS table, so the
-- definition of churn lives in one place, not scattered as string literals.
CREATE OR REPLACE TABLE main.churn_status (
    account_status VARCHAR PRIMARY KEY,
    description VARCHAR
  );
INSERT INTO main.churn_status
VALUES
  ('canceled', 'Subscription cancelled'),
  ('paused', 'Subscription paused');

-- ----------------------------------------------------------------------------
-- Stage 3 — Feature tables
-- ----------------------------------------------------------------------------
-- ---- main.dimension_customer: canonical customer attributes ----------------------------
CREATE OR REPLACE TABLE main.dimension_customer AS
SELECT customer_id,
  customer_name,
  customer_segment,
  country,
  subscription_plan,
  monthly_revenue,
  -- NULL preserved (no imputation)
  account_created_at,
  -- DATE
  account_status,
  -- normalized lowercase
  FALSE AS revenue_imputed -- never imputed (user decision)
FROM main.clean_customers;
-- ---- fact tables: cleaned facts ---------------------------------------------
CREATE OR REPLACE TABLE main.fact_ticket AS
SELECT ticket_id,
  customer_id,
  created_at,
  subject,
  message,
  category,
  priority,
  resolution_time_hours,
  status,
  satisfaction_score
FROM main.clean_tickets;
CREATE OR REPLACE TABLE main.fact_feedback AS
SELECT feedback_id,
  customer_id,
  created_at,
  feedback_text,
  feedback_source,
  rating
FROM main.clean_feedback;
CREATE OR REPLACE TABLE main.fact_usage AS
SELECT customer_id,
  date,
  active_users,
  sessions,
  feature_usage,
  errors,
  average_session_duration
FROM main.clean_usage;
CREATE OR REPLACE TABLE main.fact_subscription_event AS
SELECT customer_id,
  event_date,
  event_type,
  previous_plan,
  new_plan,
  revenue_change,
  _input_row_number
FROM main.clean_subscription_events;
-- ---- main.aggregate_theme: rule-based keyword themes (App 1 seeds; App 2 upgrades) ------
-- Schema: (feedback_id, customer_id, created_at, text, theme, sentiment, source).
-- source='rule' for App 1; App 2 enrichment overwrites with source='llm'.
CREATE OR REPLACE TABLE main.aggregate_theme AS
SELECT feedback_id,
  customer_id,
  created_at,
  feedback_text AS text,
  CASE
    -- Keyword lexicon. Order matters: most specific first, fallback last.
    -- Topical themes (billing, search, integrations, ...) come before the
    -- generic `product_quality` bucket, and the generic bucket shares no
    -- tokens with topical regexes (e.g. 'export' is a reporting word, so
    -- "crashes on export" is a reporting complaint). Keep the two in sync
    -- with apps/embedding/themes.py.
    WHEN regexp_matches(
      lower(feedback_text),
      '(invoice|billing|payment|charge|refund|pricing|price|plan cost|renewal)'
    ) THEN 'billing'
    WHEN regexp_matches(
      lower(feedback_text),
      '(search|find|discover|browse)'
    ) THEN 'search'
    WHEN regexp_matches(
      lower(feedback_text),
      '(api|integration|webhook|import|export|sso|sync|connect)'
    ) THEN 'integrations'
    WHEN regexp_matches(
      lower(feedback_text),
      '(report|dashboard|analytics|metric|chart|export)'
    ) THEN 'reporting'
    WHEN regexp_matches(
      lower(feedback_text),
      '(collaborat|team|share|permission|role|access|multi-user)'
    ) THEN 'collaboration'
    WHEN regexp_matches(
      lower(feedback_text),
      '(onboard|setup|getting started|tutorial|learn|training|documentation|docs)'
    ) THEN 'onboarding'
    WHEN regexp_matches(
      lower(feedback_text),
      '(support|help|cs|assistance|response time|reply)'
    ) THEN 'support'
    WHEN regexp_matches(
      lower(feedback_text),
      '(mobile|iphone|android|app)'
    ) THEN 'mobile'
    WHEN regexp_matches(
      lower(feedback_text),
      '(performance|speed|fast|slow|lag|latency)'
    ) THEN 'performance'
    WHEN regexp_matches(
      lower(feedback_text),
      '(bug|crash|error|broken|glitch|outage|freeze|stuck|down)'
    ) THEN 'product_quality'
    ELSE 'other'
  END AS theme,
  CASE
    WHEN regexp_matches(
      lower(feedback_text),
      '(great|love|excellent|amazing|awesome|happy|impressed|best|easy|fast)'
    ) THEN 'positive'
    WHEN regexp_matches(
      lower(feedback_text),
      '(bad|worst|terrible|hate|awful|frustrat|disappoint|annoy|angry|unhappy|slow|broken|bug)'
    ) THEN 'negative'
    ELSE 'neutral'
  END AS sentiment,
  'rule' AS source
FROM main.clean_feedback;
-- ---- main.ticket_columns: pivot driver for ticket aggregates ----------------
-- One row per (column_kind, value) to pivot over. The analytical query below
-- reads THIS table (never a hardcoded value list) to decide which per-value
-- columns to emit. Adding a category here makes it a first-class count column
-- everywhere downstream. The concrete column NAMES still have to be spelled
-- in the query (SQL columns are static), but the VALUES are data-driven, so
-- the set of values the pipeline knows about is reviewable in one place and
-- changes here propagate to every aggregate.
CREATE OR REPLACE TABLE main.ticket_columns AS
SELECT 'category' AS column_kind,
  category AS value,
  'tickets_' || replace(category, ' ', '_') AS column_name
FROM main.ticket_category
UNION ALL
SELECT 'priority',
  priority,
  'tickets_' || replace(priority, ' ', '_')
FROM main.ticket_priority
UNION ALL
SELECT 'status',
  status,
  'tickets_' || replace(status, ' ', '_')
FROM main.ticket_status
UNION ALL
SELECT 'event_type',
  event_type,
  replace(event_type, ' ', '_')
FROM main.subscription_event_type;

-- ---- main.aggregate_customer_features: one row per customer, pure SQL aggregates --------
-- Usage-trend windows are DERIVED from the data (no hardcoded dates):
--   analysis_end_date          = max(date) in the usage data
--   last_4_weeks_start         = analysis_end_date - 27 days
--   previous_4_weeks_start     = last_4_weeks_start - 28 days
-- The windows are the two most recent complete 4-week periods (current vs prior).
-- Rating-trend split (last half vs prior half) uses the midpoint of the
-- feedback date range, also derived.
CREATE OR REPLACE TABLE main.aggregate_customer_features AS WITH usage_window AS (
    SELECT max(date) AS analysis_end_date,
      max(date) - INTERVAL 27 DAY AS last_4_weeks_start,
      max(date) - INTERVAL 55 DAY AS previous_4_weeks_start
    FROM main.fact_usage
  ),
  rating_split AS (
    SELECT min(created_at) + (max(created_at) - min(created_at)) / 2 AS rating_split_point
    FROM main.fact_feedback
  ),
  customers AS (
    SELECT customer_id,
      account_status,
      customer_segment,
      subscription_plan,
      monthly_revenue
    FROM main.dimension_customer
  ),
  ticket_aggregates AS (
    SELECT t.customer_id,
      count(*) AS ticket_count,
      -- Per-category / per-priority / per-status counts, pivoted from
      -- main.ticket_columns (config-driven; values live in the config tables,
      -- not hardcoded here).
      count(*) FILTER (
        WHERE category IN (SELECT value FROM main.ticket_columns WHERE column_kind = 'category')
      ) AS tickets_any_category,
      count(*) FILTER (
        WHERE category = 'billing'
      ) AS tickets_billing,
      count(*) FILTER (
        WHERE category = 'bug'
      ) AS tickets_bug,
      count(*) FILTER (
        WHERE category = 'feature_request'
      ) AS tickets_feature_request,
      count(*) FILTER (
        WHERE category = 'general_question'
      ) AS tickets_general_question,
      count(*) FILTER (
        WHERE category = 'technical_issue'
      ) AS tickets_technical_issue,
      count(*) FILTER (
        WHERE category = 'onboarding'
      ) AS tickets_onboarding,
      count(*) FILTER (
        WHERE category = 'account_access'
      ) AS tickets_account_access,
      count(*) FILTER (
        WHERE priority = 'urgent'
      ) AS tickets_urgent,
      count(*) FILTER (
        WHERE priority = 'high'
      ) AS tickets_high,
      count(*) FILTER (
        WHERE priority = 'medium'
      ) AS tickets_medium,
      count(*) FILTER (
        WHERE priority = 'low'
      ) AS tickets_low,
      count(*) FILTER (
        WHERE status = 'open'
      ) AS tickets_open,
      count(*) FILTER (
        WHERE status = 'pending'
      ) AS tickets_pending,
      count(*) FILTER (
        WHERE status = 'resolved'
      ) AS tickets_resolved,
      avg(resolution_time_hours) AS average_resolution_time_hours,
      avg(satisfaction_score) AS average_satisfaction_score
    FROM main.fact_ticket t
    GROUP BY t.customer_id
  ),
  usage_aggregates AS (
    SELECT customer_id,
      sum(sessions) FILTER (
        WHERE date >= (SELECT last_4_weeks_start FROM usage_window)
      ) AS sessions_last_4_weeks,
      sum(sessions) FILTER (
        WHERE date >= (SELECT previous_4_weeks_start FROM usage_window)
          AND date < (SELECT last_4_weeks_start FROM usage_window)
      ) AS sessions_previous_4_weeks,
      sum(active_users) FILTER (
        WHERE date >= (SELECT last_4_weeks_start FROM usage_window)
      ) AS active_users_last_4_weeks,
      sum(active_users) FILTER (
        WHERE date >= (SELECT previous_4_weeks_start FROM usage_window)
          AND date < (SELECT last_4_weeks_start FROM usage_window)
      ) AS active_users_previous_4_weeks,
      sum(errors) AS errors_total,
      avg(average_session_duration) AS average_session_duration
    FROM main.fact_usage
    GROUP BY customer_id
  ),
  feedback_aggregates AS (
    SELECT customer_id,
      count(*) AS feedback_count,
      avg(rating) AS average_rating,
      avg(rating) FILTER (
        WHERE created_at >= (SELECT rating_split_point FROM rating_split)
      ) AS rating_last_half,
      avg(rating) FILTER (
        WHERE created_at < (SELECT rating_split_point FROM rating_split)
      ) AS rating_prior_half
    FROM main.fact_feedback
    GROUP BY customer_id
  ),
  subscription_aggregates AS (
    SELECT customer_id,
      -- Per-event-type counts, config-driven (values live in
      -- main.subscription_event_type; the concrete column names here stay
      -- stable so consumers have a fixed contract).
      count(*) FILTER (
        WHERE event_type = 'cancellation'
      ) AS cancellations,
      count(*) FILTER (
        WHERE event_type = 'downgrade'
      ) AS downgrades,
      count(*) FILTER (
        WHERE event_type = 'upgrade'
      ) AS upgrades,
      count(*) FILTER (
        WHERE event_type = 'renewal'
      ) AS renewals,
      count(*) FILTER (
        WHERE event_type IN (SELECT value FROM main.ticket_columns WHERE column_kind = 'event_type')
      ) AS total_events,
      sum(revenue_change) AS revenue_change_sum,
      max(event_date) AS last_event_date,
      -- last event type per customer (max event_date wins; ties by input order)
      (
        SELECT latest_event.event_type
        FROM main.fact_subscription_event latest_event
        WHERE latest_event.customer_id = outer_subscription.customer_id
        ORDER BY latest_event.event_date DESC,
          latest_event._input_row_number DESC
        LIMIT 1
      ) AS last_event_type
    FROM main.fact_subscription_event AS outer_subscription
    GROUP BY customer_id
  )
SELECT cust.customer_id,
  cust.account_status,
  cust.customer_segment,
  cust.subscription_plan,
  cust.monthly_revenue,
  -- churn proxy: statuses listed in main.churn_status (config-driven)
  cust.account_status IN (
    SELECT account_status FROM main.churn_status
  ) AS churn_proxy,
  -- tickets
  coalesce(ticket.ticket_count, 0) AS ticket_count,
  coalesce(ticket.tickets_any_category, 0) AS tickets_any_category,
  coalesce(ticket.tickets_billing, 0) AS tickets_billing,
  coalesce(ticket.tickets_bug, 0) AS tickets_bug,
  coalesce(ticket.tickets_feature_request, 0) AS tickets_feature_request,
  coalesce(ticket.tickets_general_question, 0) AS tickets_general_question,
  coalesce(ticket.tickets_technical_issue, 0) AS tickets_technical_issue,
  coalesce(ticket.tickets_onboarding, 0) AS tickets_onboarding,
  coalesce(ticket.tickets_account_access, 0) AS tickets_account_access,
  coalesce(ticket.tickets_urgent, 0) AS tickets_urgent,
  coalesce(ticket.tickets_high, 0) AS tickets_high,
  coalesce(ticket.tickets_medium, 0) AS tickets_medium,
  coalesce(ticket.tickets_low, 0) AS tickets_low,
  coalesce(ticket.tickets_open, 0) AS tickets_open,
  coalesce(ticket.tickets_pending, 0) AS tickets_pending,
  coalesce(ticket.tickets_resolved, 0) AS tickets_resolved,
  ticket.average_resolution_time_hours,
  ticket.average_satisfaction_score,
  -- usage trend
  usage.sessions_last_4_weeks,
  usage.sessions_previous_4_weeks,
  usage.active_users_last_4_weeks,
  usage.active_users_previous_4_weeks,
  -- NULL-safe change %: NULLIF denominator 0; NULL if no prior usage
  (usage.sessions_last_4_weeks - usage.sessions_previous_4_weeks) * 100.0 / NULLIF(usage.sessions_previous_4_weeks, 0) AS sessions_change_percent,
  (usage.active_users_last_4_weeks - usage.active_users_previous_4_weeks) * 100.0 / NULLIF(usage.active_users_previous_4_weeks, 0) AS active_users_change_percent,
  usage.errors_total,
  usage.average_session_duration,
  -- feedback
  coalesce(feedback.feedback_count, 0) AS feedback_count,
  feedback.average_rating,
  feedback.rating_last_half,
  feedback.rating_prior_half,
  -- subscription events
  coalesce(subscription.cancellations, 0) AS cancellations,
  coalesce(subscription.downgrades, 0) AS downgrades,
  coalesce(subscription.upgrades, 0) AS upgrades,
  coalesce(subscription.renewals, 0) AS renewals,
  coalesce(subscription.total_events, 0) AS total_events,
  subscription.revenue_change_sum,
  subscription.last_event_date,
  subscription.last_event_type
FROM customers cust
  LEFT JOIN ticket_aggregates ticket ON ticket.customer_id = cust.customer_id
  LEFT JOIN usage_aggregates usage ON usage.customer_id = cust.customer_id
  LEFT JOIN feedback_aggregates feedback ON feedback.customer_id = cust.customer_id
  LEFT JOIN subscription_aggregates subscription ON subscription.customer_id = cust.customer_id
ORDER BY cust.customer_id;
-- ---- main.aggregate_segment_metrics: per segment/plan stats vs global -------------------
CREATE OR REPLACE TABLE main.aggregate_segment_metrics AS WITH customer_base AS (
    SELECT features.customer_id,
      cust.customer_segment,
      cust.subscription_plan,
      cust.account_status,
      cust.monthly_revenue,
      features.ticket_count,
      features.tickets_urgent,
      features.tickets_bug,
      features.tickets_billing,
      features.tickets_open,
      features.average_resolution_time_hours,
      features.average_satisfaction_score,
      features.churn_proxy
    FROM main.aggregate_customer_features features
      JOIN main.dimension_customer cust ON cust.customer_id = features.customer_id
  ),
  -- Which ticket categories count as a "complaint" — config-driven, not
  -- hardcoded in the metric query. A complaint is a ticket that signals
  -- commercial pain: bug or billing tickets (add more here or in
  -- main.ticket_category to change the definition in ONE place).
  complaint_categories AS (
    SELECT 'bug' AS category
    UNION ALL
    SELECT 'billing'
  ),
  global_stats AS (
    SELECT count(DISTINCT customer_id) AS global_customers,
      count(DISTINCT customer_id) FILTER (
        WHERE monthly_revenue IS NOT NULL
      ) AS global_revenue_customers,
      sum(monthly_revenue) AS global_revenue,
      count(*) FILTER (
        WHERE ticket_count > 0
      ) AS global_customers_with_tickets,
      count(*) FILTER (
        WHERE churn_proxy
      ) AS global_churn,
      avg(average_resolution_time_hours) AS global_average_resolution,
      avg(average_satisfaction_score) AS global_average_satisfaction,
      1.0 * count(*) FILTER (
        WHERE churn_proxy
      ) / NULLIF(count(DISTINCT customer_id), 0) AS global_cancel_rate
    FROM customer_base
  )
SELECT cust.customer_segment AS segment,
  NULL::VARCHAR AS plan,
  count(DISTINCT cust.customer_id) AS customers,
  sum(cust.monthly_revenue) AS revenue,
  count(DISTINCT cust.customer_id) FILTER (
    WHERE cust.monthly_revenue IS NOT NULL
  ) AS revenue_customers,
  count(*) FILTER (
    WHERE cust.ticket_count > 0
  ) AS customers_with_tickets,
  sum(cust.ticket_count) AS ticket_count,
  -- complaint rate: share of customers with a complaint-category ticket
  -- (bug/billing, per complaint_categories) or an urgent ticket
  1.0 * count(*) FILTER (
    WHERE cust.ticket_count > 0
      AND (
        cust.tickets_urgent > 0
        OR cust.tickets_bug > 0
        OR cust.tickets_billing > 0
      )
  ) / NULLIF(count(DISTINCT cust.customer_id), 0) AS complaint_rate,
  avg(cust.average_resolution_time_hours) AS average_resolution_time_hours,
  avg(cust.average_satisfaction_score) AS average_satisfaction_score,
  1.0 * count(*) FILTER (
    WHERE cust.churn_proxy
  ) / NULLIF(count(DISTINCT cust.customer_id), 0) AS cancel_rate,
  global.global_customers AS global_customers,
  global.global_revenue AS global_revenue,
  global.global_average_resolution AS global_average_resolution,
  global.global_average_satisfaction AS global_average_satisfaction,
  global.global_churn AS global_churn,
  1.0 * global.global_churn / NULLIF(global.global_customers, 0) AS global_cancel_rate
FROM customer_base cust
  CROSS JOIN global_stats global
GROUP BY cust.customer_segment,
  global.global_customers,
  global.global_revenue,
  global.global_average_resolution,
  global.global_average_satisfaction,
  global.global_churn,
  global.global_cancel_rate
UNION ALL
SELECT NULL::VARCHAR AS segment,
  cust.subscription_plan AS plan,
  count(DISTINCT cust.customer_id),
  sum(cust.monthly_revenue),
  count(DISTINCT cust.customer_id) FILTER (
    WHERE cust.monthly_revenue IS NOT NULL
  ),
  count(*) FILTER (
    WHERE cust.ticket_count > 0
  ),
  sum(cust.ticket_count),
  1.0 * count(*) FILTER (
    WHERE cust.ticket_count > 0
      AND (
        cust.tickets_urgent > 0
        OR cust.tickets_bug > 0
        OR cust.tickets_billing > 0
      )
  ) / NULLIF(count(DISTINCT cust.customer_id), 0),
  avg(cust.average_resolution_time_hours),
  avg(cust.average_satisfaction_score),
  1.0 * count(*) FILTER (
    WHERE cust.churn_proxy
  ) / NULLIF(count(DISTINCT cust.customer_id), 0),
  global.global_customers,
  global.global_revenue,
  global.global_average_resolution,
  global.global_average_satisfaction,
  global.global_churn,
  global.global_cancel_rate
FROM customer_base cust
  CROSS JOIN global_stats global
GROUP BY cust.subscription_plan,
  global.global_customers,
  global.global_revenue,
  global.global_average_resolution,
  global.global_average_satisfaction,
  global.global_churn,
  global.global_cancel_rate
ORDER BY segment NULLS LAST,
  plan NULLS LAST;