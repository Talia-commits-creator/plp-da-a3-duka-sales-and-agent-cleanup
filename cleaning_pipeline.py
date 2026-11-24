
import pandas as pd


def clean_agent_data(txns, agents):
    """
    Clean agent transactions and generate a data-quality audit log.
    Returns the cleaned DataFrame and audit log.
    """

    clean_df = txns.copy()
    audit = []

    def add_log(issue, dimension, column, rows_affected):
        audit.append({
            "issue": issue,
            "dimension": dimension,
            "column": column,
            "rows_affected": int(rows_affected)
        })

    # Remove duplicate transaction IDs
    duplicate_count = clean_df["txn_id"].duplicated().sum()
    clean_df = clean_df.drop_duplicates(
        subset="txn_id", keep="first"
    ).copy()

    add_log("Duplicate transaction IDs removed",
            "Uniqueness", "txn_id", duplicate_count)

    # Convert amounts to numeric
    clean_df["amount"] = pd.to_numeric(
        clean_df["amount"], errors="coerce"
    )

    # Flag missing amounts
    clean_df["amount_imputed"] = clean_df["amount"].isna()
    missing_count = clean_df["amount_imputed"].sum()

    # Detect outliers using IQR
    q1 = clean_df["amount"].quantile(0.25)
    q3 = clean_df["amount"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    clean_df["is_outlier"] = (
        (clean_df["amount"] < lower_bound) |
        (clean_df["amount"] > upper_bound)
    )

    outlier_count = clean_df["is_outlier"].sum()

    # Impute missing amounts with median
    clean_df["amount"] = clean_df["amount"].fillna(
        clean_df["amount"].median()
    )

    add_log("Missing amounts filled with median",
            "Completeness", "amount", missing_count)

    add_log("Unusual amounts flagged using IQR",
            "Accuracy", "amount", outlier_count)

    # Standardize transaction types
    original_types = clean_df["txn_type"].copy()

    clean_df["txn_type"] = (
        clean_df["txn_type"].astype("string").str.strip().str.title()
    )

    standardized_count = (
        original_types != clean_df["txn_type"]
    ).sum()

    add_log("Transaction types standardized",
            "Consistency", "txn_type", standardized_count)

    # Standardize dates
    original_dates = clean_df["txn_date"].astype("string")

    clean_df["txn_date"] = pd.to_datetime(
        original_dates.str.replace("/", "-", regex=False),
        errors="coerce"
    )

    invalid_dates = clean_df["txn_date"].isna().sum()

    changed_dates = (
        original_dates != clean_df["txn_date"].dt.strftime("%Y-%m-%d")
    ).sum()

    clean_df["txn_date"] = clean_df["txn_date"].dt.strftime("%Y-%m-%d")

    add_log("Transaction dates standardized",
            "Consistency", "txn_date", changed_dates)

    add_log("Invalid transaction dates",
            "Validity", "txn_date", invalid_dates)

    # Merge county information
    clean_df = clean_df.merge(
        agents[["agent_id", "county"]],
        on="agent_id",
        how="left",
        validate="many_to_one"
    )

    unknown_count = clean_df["county"].isna().sum()
    clean_df["county"] = clean_df["county"].fillna("Unknown")

    add_log("Transactions linked to unregistered agents",
            "Validity", "agent_id", unknown_count)

    log = pd.DataFrame(
        audit,
        columns=["issue", "dimension", "column", "rows_affected"]
    )

    return clean_df, log
