from __future__ import annotations

from typing import Any

import pandas as pd


def _severity_from_rate(rate: float) -> str:
    if rate >= 20:
        return "High"
    if rate >= 8:
        return "Medium"
    return "Low"


def _add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    column: str,
    concern: str,
    finding: str,
    recommendation: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "column": column,
            "concern": concern,
            "finding": finding,
            "recommendation": recommendation,
        }
    )


def review_hygiene(df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].replace(r"^\s*$", pd.NA, regex=True)

    duplicate_rows = int(cleaned.duplicated().sum())
    if duplicate_rows:
        _add_issue(
            issues,
            "Medium",
            "Dataset",
            "Duplicate rows",
            f"{duplicate_rows} rows are exact duplicates and could make the synthetic output overfit.",
            "Remove or deduplicate repeated encounters before using the dataset in production.",
        )

    for column, details in profile["columns"].items():
        missing_pct = float(details["missing_pct"])
        if missing_pct > 0:
            _add_issue(
                issues,
                _severity_from_rate(missing_pct),
                column,
                "Missingness",
                f"{missing_pct:.1f}% of values are missing.",
                "Keep nullable fields explicit in metadata and consider whether missing values should be sampled or suppressed.",
            )

        if details["semantic_role"] == "identifier":
            _add_issue(
                issues,
                "Medium",
                column,
                "Identifier handling",
                "This field behaves like an encounter-level identifier.",
                "Generate surrogate values instead of carrying source identifiers into the synthetic output.",
            )

        if details["semantic_role"] == "numeric":
            numeric = pd.to_numeric(cleaned[column], errors="coerce").dropna()
            if numeric.empty:
                continue

            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                low_cutoff = q1 - 1.5 * iqr
                high_cutoff = q3 + 1.5 * iqr
                outlier_count = int(((numeric < low_cutoff) | (numeric > high_cutoff)).sum())
                if outlier_count:
                    _add_issue(
                        issues,
                        "Medium" if outlier_count <= 2 else "High",
                        column,
                        "Outliers",
                        f"{outlier_count} values fall outside the IQR-based expected range.",
                        "Decide whether these extremes should be preserved, clipped, or modelled as a scenario.",
                    )

            lowered = column.lower()
            if "wait" in lowered:
                extreme_waits = int((numeric > 360).sum())
                if extreme_waits:
                    _add_issue(
                        issues,
                        "High" if extreme_waits >= 2 else "Medium",
                        column,
                        "Extreme wait times",
                        f"{extreme_waits} encounters exceed 360 minutes of waiting time.",
                        "Use the scenario control to compress the longest waits while preserving overall flow patterns.",
                    )
            if any(token in lowered for token in ["age", "wait", "stay"]) and int((numeric < 0).sum()):
                _add_issue(
                    issues,
                    "High",
                    column,
                    "Negative values",
                    "Negative values appear in a field that should be non-negative.",
                    "Clean or cap invalid negative values before synthetic generation.",
                )

        if details["semantic_role"] == "date":
            parsed = pd.to_datetime(cleaned[column], errors="coerce", format="mixed")
            invalid_dates = int((cleaned[column].notna() & parsed.isna()).sum())
            if invalid_dates:
                _add_issue(
                    issues,
                    "Medium",
                    column,
                    "Invalid dates",
                    f"{invalid_dates} values do not parse as valid dates.",
                    "Convert unparseable dates to missing or standardize them before approval.",
                )

        if details["semantic_role"] in {"categorical", "binary"}:
            non_null = cleaned[column].dropna().astype(str)
            if non_null.empty:
                continue
            normalized = non_null.str.strip().str.lower()
            if normalized.nunique() < non_null.nunique():
                _add_issue(
                    issues,
                    "Low",
                    column,
                    "Category normalization",
                    "Some category labels differ only by casing or spacing.",
                    "Standardize category labels to avoid splitting one concept into multiple buckets.",
                )

    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    issues.sort(key=lambda item: severity_order.get(item["severity"], 99))

    severity_counts = {"High": 0, "Medium": 0, "Low": 0}
    for issue in issues:
        severity_counts[issue["severity"]] += 1

    quality_score = max(0, 100 - severity_counts["High"] * 12 - severity_counts["Medium"] * 6 - severity_counts["Low"] * 2)

    return {
        "issues": issues,
        "severity_counts": severity_counts,
        "quality_score": quality_score,
        "summary": {
            "issues_found": len(issues),
            "high_priority": severity_counts["High"],
            "duplicate_rows": duplicate_rows,
        },
    }
