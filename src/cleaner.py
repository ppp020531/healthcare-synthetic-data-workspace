from __future__ import annotations

from typing import Any

import pandas as pd


def _standardize_blank_strings(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = df.copy()
    converted = 0
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            blank_mask = cleaned[column].astype(str).str.match(r"^\s*$", na=False)
            converted += int(blank_mask.sum())
            cleaned.loc[blank_mask, column] = pd.NA
    return cleaned, converted


def _normalize_category_labels(series: pd.Series) -> tuple[pd.Series, int]:
    if series.dtype != object:
        return series, 0

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return series, 0

    normalized = non_null.str.strip().str.lower()
    groups: dict[str, list[str]] = {}
    for original, key in zip(non_null, normalized):
        groups.setdefault(key, []).append(original)

    replacement_map: dict[str, str] = {}
    normalized_changes = 0
    for values in groups.values():
        canonical = pd.Series(values).value_counts().idxmax()
        for value in set(values):
            replacement_map[value] = canonical
            if value != canonical:
                normalized_changes += 1

    updated = series.map(lambda value: replacement_map.get(value, value) if pd.notna(value) else value)
    return updated, normalized_changes


def _looks_like_date(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.6 and non_null.nunique() > 2)


def _group_rare_labels(series: pd.Series) -> tuple[pd.Series, int]:
    if series.dtype != object:
        return series, 0

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return series, 0

    threshold = max(2, int(round(len(non_null) * 0.03)))
    counts = non_null.value_counts()
    rare_values = set(counts[counts <= threshold].index)
    if not rare_values:
        return series, 0

    updated = series.map(lambda value: "Other" if pd.notna(value) and str(value) in rare_values else value)
    changed = int(non_null.isin(rare_values).sum())
    return updated, changed


def _fill_operational_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    filled = df.copy()
    changed = 0
    identifier_tokens = {"encounter_id", "patient_id", "visit_id", "mrn"}
    for column in filled.columns:
        lowered = column.lower()
        if any(token in lowered for token in identifier_tokens):
            continue

        series = filled[column]
        missing_mask = series.isna()
        missing_count = int(missing_mask.sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            non_null = numeric.dropna()
            if non_null.empty:
                continue
            filled.loc[missing_mask, column] = float(non_null.median())
            changed += missing_count
            continue

        if _looks_like_date(series):
            continue

        filled.loc[missing_mask, column] = "Unknown"
        changed += missing_count

    return filled, changed


def _cap_numeric_extremes(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    capped = df.copy()
    changed = 0
    for column in capped.columns:
        numeric = pd.to_numeric(capped[column], errors="coerce")
        non_null = numeric.dropna()
        if non_null.empty or non_null.nunique() <= 3:
            continue
        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clipped = numeric.clip(lower=lower, upper=upper)
        diff_mask = numeric.notna() & clipped.ne(numeric)
        if bool(diff_mask.any()):
            capped.loc[diff_mask, column] = clipped.loc[diff_mask]
            changed += int(diff_mask.sum())
    return capped, changed


def _repair_invalid_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    repaired = df.copy()
    changed = 0
    for column in repaired.columns:
        series = repaired[column]
        if pd.api.types.is_numeric_dtype(series) or not _looks_like_date(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        invalid_mask = series.notna() & parsed.isna()
        if bool(invalid_mask.any()):
            repaired.loc[invalid_mask, column] = pd.NA
            changed += int(invalid_mask.sum())
    return repaired, changed


def apply_hygiene_fixes(
    df: pd.DataFrame,
    options: dict[str, bool],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cleaned = df.copy()
    actions: list[dict[str, Any]] = []

    if options.get("standardize_blank_strings", False):
        cleaned, converted = _standardize_blank_strings(cleaned)
        actions.append(
            {
                "action": "Standardize blank strings",
                "effect": (
                    f"Converted {converted} blank cells to missing values."
                    if converted
                    else "No blank-string cleanup was needed."
                ),
            }
        )

    if options.get("remove_duplicates", False):
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        removed = before - len(cleaned)
        actions.append(
            {
                "action": "Remove duplicate rows",
                "effect": f"Removed {removed} duplicate rows." if removed else "No duplicate rows were found.",
            }
        )

    if options.get("normalize_categories", False):
        total_changes = 0
        for column in cleaned.columns:
            updated, changes = _normalize_category_labels(cleaned[column])
            cleaned[column] = updated
            total_changes += changes
        actions.append(
            {
                "action": "Normalize category labels",
                "effect": (
                    f"Standardized {total_changes} category value variations."
                    if total_changes
                    else "No category normalization issues were found."
                ),
            }
        )

    if options.get("fill_operational_gaps", False):
        cleaned, filled = _fill_operational_gaps(cleaned)
        actions.append(
            {
                "action": "Fill common operational gaps",
                "effect": (
                    f"Filled {filled} missing cells using simple operational defaults."
                    if filled
                    else "No eligible operational gaps were filled."
                ),
            }
        )

    if options.get("fix_negative_values", False):
        fixed_columns: list[str] = []
        for column in cleaned.columns:
            lowered = column.lower()
            if any(token in lowered for token in ["age", "wait", "stay", "ctas"]):
                numeric = pd.to_numeric(cleaned[column], errors="coerce")
                negative_mask = numeric < 0
                if bool(negative_mask.fillna(False).any()):
                    cleaned.loc[negative_mask, column] = pd.NA
                    fixed_columns.append(column)
        actions.append(
            {
                "action": "Correct negative operational values",
                "effect": (
                    "Converted negative values to missing in: " + ", ".join(fixed_columns)
                    if fixed_columns
                    else "No invalid negative values were found."
                ),
            }
        )

    if options.get("repair_invalid_dates", False):
        cleaned, repaired = _repair_invalid_dates(cleaned)
        actions.append(
            {
                "action": "Repair invalid dates",
                "effect": (
                    f"Converted {repaired} invalid date values to missing."
                    if repaired
                    else "No invalid date values were found."
                ),
            }
        )

    if options.get("cap_numeric_extremes", False):
        cleaned, capped = _cap_numeric_extremes(cleaned)
        actions.append(
            {
                "action": "Cap numeric extremes",
                "effect": (
                    f"Trimmed {capped} extreme numeric values to bounded ranges."
                    if capped
                    else "No numeric extremes required capping."
                ),
            }
        )

    if options.get("group_rare_categories", False):
        total_grouped = 0
        for column in cleaned.columns:
            updated, changes = _group_rare_labels(cleaned[column])
            cleaned[column] = updated
            total_grouped += changes
        actions.append(
            {
                "action": "Group rare categories",
                "effect": (
                    f"Grouped {total_grouped} low-frequency labels into 'Other'."
                    if total_grouped
                    else "No rare categories required grouping."
                ),
            }
        )

    return cleaned, actions
