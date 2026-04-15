# Healthcare Synthetic Data Workspace
## Methodology and Algorithm Logic for Judge Review

This document explains what the system does, how each algorithm works, how conclusions are produced, how comparisons are made, and why the design choices are reasonable for a healthcare synthetic data demo.

The goal of the platform is not to claim perfect privacy or perfect realism. The goal is to provide a transparent, controllable, and explainable workflow that helps hospital teams transform sensitive healthcare data into safer synthetic data while preserving enough structure for analytics, testing, and prototyping.

---

## 1. End-to-End Logic

The platform follows a simple operational chain:

1. Upload source data
2. Scan the data for quality and sensitivity issues
3. Review and adjust metadata controls
4. Generate synthetic data from the approved metadata package
5. Validate fidelity and privacy risk
6. Release the synthetic output in a controlled way

The key design principle is:

**Raw data -> structured metadata -> controlled synthetic generation -> measurable validation**

That is important because it avoids a black-box "upload and magically generate" workflow. Every major transformation is surfaced and reviewable.

---

## 2. What the System Does at Each Stage

### 2.1 Data Profiling

The first algorithmic step is profiling the uploaded dataset.

What it does:

- Replaces blank strings with missing values
- Examines each column independently
- Infers the column's semantic role
- Summarizes missingness, uniqueness, value examples, and basic statistics

The system classifies each column into one of these roles:

- `identifier`
- `numeric`
- `categorical`
- `binary`
- `date`

How role inference works:

- If the column name looks like `encounter_id`, `patient_id`, `visit_id`, or `mrn`, it is treated as an identifier
- Boolean types become binary
- Numeric columns become numeric unless they only have two unique values, in which case they are treated as binary
- Non-numeric columns are tested for date parseability; if at least 85% of non-null values parse as dates and there are more than 2 unique values, the column is treated as a date
- If a non-null text column only has two normalized values, it is treated as binary
- Otherwise it is categorical

Why this is reasonable:

- It is transparent
- It uses both schema hints and observed values
- It is robust enough for operational hospital files without requiring a rigid external schema

What the profiler outputs:

- dataset shape
- total missing cells
- duplicate rows
- number of columns by role
- per-column summaries such as missing rate, unique rate, numeric summary stats, or top categories

This step gives the rest of the workflow a structured understanding of the source data before any synthetic generation begins.

---

### 2.2 Hygiene Review

The hygiene review scans the profiled data for practical data issues that matter for synthetic generation quality and privacy safety.

The system looks for:

- duplicate rows
- missingness
- identifier-like fields
- numeric outliers
- invalid negative values in operational fields
- invalid dates
- category normalization issues

#### Missingness logic

For every column with missing values, the system creates an issue and assigns severity by rate:

- `High` if missingness >= 20%
- `Medium` if missingness >= 8%
- `Low` otherwise

Why this is reasonable:

- Missingness directly affects analytic utility
- A transparent threshold model is easy to explain
- The point is not perfect statistical certification; it is operational risk visibility

#### Duplicate detection

The system counts exact duplicate rows.

Why this matters:

- Duplicate rows can cause synthetic overfitting
- They can distort distributions
- They are often a sign of ingestion or merge issues

#### Identifier detection

Any column classified as an identifier is flagged for surrogate replacement.

Why this matters:

- Direct identifiers should never pass into a synthetic release unchanged
- Even in a demo, it is important to show that identifiers are handled explicitly

#### Outlier detection

For numeric columns, the system uses the standard IQR rule:

- `Q1 = 25th percentile`
- `Q3 = 75th percentile`
- `IQR = Q3 - Q1`
- outliers are values below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR`

Severity logic:

- `Medium` if there are only a small number of outliers
- `High` if the count is larger

Why this is reasonable:

- The IQR rule is widely understood
- It is explainable to non-technical audiences
- It identifies unusual tails without assuming normality

#### Invalid negative values

For operational fields such as age, wait, stay, or CTAS-related numeric fields, any negative values are flagged as high-priority issues.

Why this is reasonable:

- These values violate basic business logic
- They should not be silently propagated into synthetic output

#### Invalid dates

Date-like columns are parsed, and any non-null value that fails to parse is flagged.

Why this is reasonable:

- Bad dates can break downstream timelines
- Invalid timestamps can create misleading synthetic sequences

#### Category normalization issues

If category values differ only by case or spacing, the system flags them.

Example:

- `Admitted`
- `admitted`
- `Admitted `

These should not become separate categories.

#### Hygiene quality score

The hygiene score is a simple weighted heuristic:

`Quality Score = max(0, 100 - 12 * High - 6 * Medium - 2 * Low)`

Why this is reasonable:

- It is intentionally simple
- It penalizes high-severity issues more heavily
- It gives judges a readable quality indicator without pretending to be a regulatory standard

This score is not a clinical certification. It is an operational data-cleanliness indicator.

---

### 2.3 Hygiene Fixes

The platform can optionally apply targeted cleanup actions before metadata approval.

Supported actions include:

- standardize blank strings
- remove duplicate rows
- normalize category labels
- fill common operational gaps
- correct negative operational values
- repair invalid dates
- cap numeric extremes
- group rare categories

#### Blank string standardization

Blank text cells are converted to missing values.

Why this is reasonable:

- Empty strings and true missing values should not be treated differently by accident

#### Category normalization

The system lowercases and trims values to identify equivalent labels, then chooses the most frequent real spelling as the canonical label.

Why this is reasonable:

- It preserves the dominant operational spelling
- It reduces accidental category fragmentation

#### Filling operational gaps

This is intentionally conservative:

- numeric fields are filled with the median
- non-date text fields are filled with `"Unknown"`
- identifiers are never filled
- likely date columns are not auto-filled

Why this is reasonable:

- Median is stable for operational numeric fields
- `"Unknown"` keeps categorical gaps explicit
- Skipping identifiers and dates avoids overconfident imputation

#### Capping numeric extremes

This uses the same IQR boundaries and clips values to the lower and upper fences.

Why this is reasonable:

- It reduces the influence of implausible extremes
- It is safer than deleting rows

#### Rare category grouping

Rare labels are grouped into `"Other"` if their frequency is very low:

- threshold = max(2, 3% of non-null rows)

Why this is reasonable:

- Very sparse categories can leak uniqueness
- Grouping them improves stability and privacy

---

### 2.4 Metadata Construction and Control

This is one of the most important parts of the system.

Instead of generating synthetic data directly from raw fields, the system first converts the source schema into a metadata package. Each field gets:

- inclusion decision
- inferred data type
- generation strategy
- control action
- nullability flag
- explanatory note

Default generation strategies:

- identifier -> `new_token`
- numeric -> `sample_plus_noise`
- categorical -> `sample_category`
- binary -> `sample_category`
- date -> `sample_plus_jitter`

Default control actions:

- identifiers -> `Tokenize`
- dates -> `Date shift`
- postal/address fields -> `Coarse geography`
- complaint/note/text fields -> `Group text`
- numeric fields -> `Preserve`

Why this is reasonable:

- Different field types require different privacy treatments
- The rules are transparent and editable
- It gives a human-readable bridge between raw data and synthetic output

#### Sensitivity classification

The app also classifies fields into governance levels:

- `Restricted`: direct identifiers
- `Sensitive`: dates, postal/location, free text
- `Operational`: everything else

Why this is reasonable:

- Not all fields carry equal privacy risk
- It helps explain why some fields need stronger controls than others

#### Ownership logic

Field ownership is also assigned:

- `Manager / Reviewer` owns sensitive and restricted fields
- `Data Analyst` owns operational fields

Why this is reasonable:

- Sensitive handling should not depend only on the analyst
- It makes governance visible without making the workflow overly complex

---

## 3. Synthetic Data Generation Logic

The generator produces synthetic records using pragmatic heuristics rather than a heavy black-box generative model.

This is intentional.

For a hospital demo, the priorities are:

- transparency
- field-level control
- understandable privacy tradeoffs
- predictable behavior

The generator uses the following controls:

- privacy vs fidelity balance
- synthetic row count
- locked key distributions
- correlation preservation
- rare case retention
- noise level
- missingness pattern
- outlier strategy
- field-level metadata actions

### 3.1 Anchor-row sampling

Before generating fields, the system chooses anchor rows from the source dataset.

These anchors are sampled using a weighted mixture:

- uniform row sampling
- rare-row weighted sampling

Rare-row weights are higher for:

- low-frequency categorical combinations
- numeric tail observations

The final anchor sampling mixture increases rare-row influence when `rare_case_retention` is higher.

Why this is reasonable:

- Rare but meaningful operational cases should not disappear
- It improves realism for edge cases without directly copying rows

---

### 3.2 Identifier generation

Identifier-like fields are never copied.

Instead, the system generates synthetic surrogate tokens such as:

- prefix from the column name
- sequential synthetic numbering

Why this is reasonable:

- It preserves schema familiarity
- It ensures direct identifiers are not reused

---

### 3.3 Numeric generation

Numeric fields are generated by:

1. bootstrap-sampling source values
2. adding bounded Gaussian noise
3. optionally clipping or smoothing tails
4. blending with anchor structure depending on correlation settings

#### Noise logic

Noise strength is determined by:

- fidelity priority
- noise level
- whether the column is distribution-locked

Higher fidelity -> less noise  
Higher noise level -> more perturbation  
Locked distributions -> less noise

Why this is reasonable:

- It lets the user tune privacy vs realism
- It avoids copying source values exactly
- It preserves the broad operational shape of the distribution

#### Outlier strategies

Three options exist:

- `Preserve tails`
- `Clip extremes`
- `Smooth tails`

`Clip extremes` uses 5th and 95th percentile bounds.  
`Smooth tails` compresses only the extreme ends rather than hard-cutting them.

Why this is reasonable:

- Different downstream use cases tolerate outliers differently
- Some teams want realistic tails; others want more conservative outputs

#### Non-negative enforcement

Fields like age, wait, stay, and CTAS-related values are clipped at zero if needed.

Why this is reasonable:

- These fields have known operational bounds
- Prevents synthetic nonsense values

#### Integer preservation

If the original source column is integer-like, the synthetic result is rounded.

Why this is reasonable:

- Many operational fields are count-like
- Keeps outputs natural and easier to use downstream

---

### 3.4 Categorical generation

Categorical generation starts from the empirical category distribution, then applies smoothing.

The system:

1. computes normalized category probabilities
2. blends them with a uniform distribution
3. optionally boosts rare categories
4. samples new values from the adjusted distribution

#### Smoothing logic

Smoothing is stronger when:

- fidelity is lower
- noise level is higher

Smoothing is weaker when:

- the distribution is locked

Why this is reasonable:

- Without smoothing, small datasets can overfit exact frequencies
- With too much smoothing, useful structure is lost
- This gives a practical middle ground

#### Rare case retention

Low-frequency categories can be boosted when rare-case retention is increased.

Why this is reasonable:

- Rare operational categories may be clinically or operationally important
- Keeping some representation of them improves downstream utility

---

### 3.5 Date generation

Date fields are generated by:

1. bootstrap-sampling source dates
2. adding day-level jitter
3. optionally converting to month-only output

Jitter size depends on:

- fidelity priority
- noise level
- locked distribution setting

Why this is reasonable:

- Exact encounter timing is a common quasi-identifier
- Small date shifts protect privacy while preserving broad timeline behavior

Month-only mode is even more conservative and is useful when exact day resolution is unnecessary.

---

### 3.6 Missingness handling

The generator can handle missingness in three ways:

- `Preserve source pattern`
- `Reduce gaps`
- `Fill gaps`

#### Preserve source pattern

The generator keeps missingness aligned to anchor rows when possible.

Why this is reasonable:

- Missingness is often analytically meaningful
- Keeping it helps preserve realistic data sparsity

#### Reduce gaps

The generator keeps missingness but reduces its frequency using a multiplier.

Why this is reasonable:

- Useful when source missingness is operational noise rather than intended signal

#### Fill gaps

The generator emits no synthetic missing mask for that field.

Why this is reasonable:

- Some testing environments need complete records

---


### 3.7 Correlation preservation

After generating a field, the system can blend part of the output back toward the anchor-row values.

Blend strength increases with:

- higher correlation preservation
- locked distributions

Why this is reasonable:

- Independent column sampling can destroy row-level structure
- Controlled anchor blending preserves some multivariate realism without simply copying records

This is not a full causal or probabilistic dependency model. It is a transparent approximation that improves realism in a controlled way.

---

## 4. PHI Detection and Control Logic

The platform surfaces likely PHI and quasi-identifiers before release.

Fields are flagged as:

- direct identifiers
- date/timing fields
- geographic quasi-identifiers
- free-text clinical context

For each flagged field, the system explains:

- why it is risky
- what control is applied

Examples:

- identifiers -> replaced with surrogate tokens
- exact dates -> jittered
- postal/location fields -> reduced to coarse geography
- free text -> normalized or grouped

Why this is reasonable:

- Judges and hospital teams want to see that privacy is operationalized, not just claimed
- The app makes the control point visible per field

---

## 5. How Validation Works

Validation answers two questions:

1. Is the synthetic data still useful?
2. Is the privacy posture acceptable for the current settings?

The platform does not rely on one score alone. It uses several complementary measures.

### 5.1 Column-level fidelity scoring

Each included field gets its own fidelity score.

#### Numeric fields

For numeric columns, the system compares:

- mean
- standard deviation
- 25th percentile
- median
- 75th percentile

The differences are normalized and averaged into a penalty.

Then:

`Numeric score = max(0, 1 - average penalty) * 100`

Why this is reasonable:

- Mean alone is not enough
- Spread and quantiles matter for operational realism
- This is simple, interpretable, and robust enough for demo use

#### Categorical fields

For categorical columns, the system compares the normalized source and synthetic distributions using total variation distance:

`TVD = 0.5 * sum(|p_source - p_synthetic|)`

Then:

`Categorical score = max(0, 1 - TVD) * 100`

Why this is reasonable:

- TVD is intuitive and bounded
- It directly measures distribution drift

### 5.2 Fidelity score

The overall fidelity score is the average of all per-column scores.

Why this is reasonable:

- It gives a dataset-level view while still allowing column-level drilldown

### 5.3 Privacy score

The privacy score combines three signals:

1. exact overlap of non-identifier row signatures
2. identifier reuse count
3. privacy pressure from the fidelity slider

The formula is:

`Privacy score = 100 - exact_overlap_rate * 100 - identifier_overlap * 10 - privacy_pressure`

Where:

- `exact_overlap_rate` compares row signatures across active non-identifier, non-date columns
- `identifier_overlap` counts any reused identifiers
- `privacy_pressure = (fidelity_priority / 100) * 10`

Why this is reasonable:

- It penalizes the clearest privacy risks first
- It makes the privacy-vs-fidelity tradeoff explicit
- It is easy to explain to judges

Important note:

This is a heuristic privacy score, not formal differential privacy.

That is an honest and important distinction.

### 5.4 Overall score

The overall score combines fidelity and privacy:

`Overall score = 0.6 * fidelity + 0.4 * privacy`

Why this weighting is reasonable:

- The product is meant to produce usable synthetic data, not only maximize privacy
- Utility should matter slightly more than privacy in a synthetic-data sandbox, but privacy still carries substantial weight

### 5.5 Verdict

The app translates the score into a human-readable verdict:

- `>= 85`: strong demo-ready baseline
- `>= 70`: usable baseline with review recommended
- `< 70`: needs another pass

Why this is reasonable:

- Judges do not want raw numbers only
- A short interpretation helps non-technical stakeholders understand the implications

---

## 6. How the App Compares Source vs Synthetic Data

The platform provides field-specific comparison views.

### Numeric comparison

For numeric fields:

- source and synthetic values are combined
- the combined range is split into 12 bins
- source and synthetic percentages are computed in each bin
- medians and means are also summarized

Why this is reasonable:

- It shows distribution shape rather than only one statistic
- It makes drift visible at a glance

### Date comparison

For date fields:

- the app compares daily or weekly periods depending on timeline length
- source and synthetic percentages are shown over time

Why this is reasonable:

- It preserves a readable temporal story
- It avoids overfocusing on exact dates

### Categorical comparison

For categorical fields:

- the app shows the top 8 categories by combined frequency
- source and synthetic percentages are compared side by side

Why this is reasonable:

- It focuses attention on the categories that matter most
- It avoids noisy long-tail plots

---

## 7. Dashboard Metrics Beyond the Core Validation Scores

The app also shows some higher-level summary metrics.

### Schema match

Measures how many approved fields appear in the synthetic output:

`Schema match = retained approved fields / total approved fields * 100`

Why this matters:

- Synthetic data is less useful if the schema is incomplete

### Statistical fidelity

Blends fidelity and schema coverage:

`Statistical fidelity = 0.75 * fidelity_score + 0.25 * schema_match`

Why this matters:

- A strong score should reflect both realistic values and retained structure

### Downstream utility

This is a heuristic metric for practical usefulness:

`Downstream utility = 0.62 * overall_score + 0.18 * schema_match + 0.20 * unresolved_hygiene_pressure`

Where hygiene pressure is derived from unresolved hygiene severity.

Why this matters:

- A synthetic dataset can look statistically good but still be hard to use if hygiene is poor or schema coverage is weak
- This metric makes the evaluation more operational

Again, this is a practical heuristic, not a formal regulatory standard.

---

## 8. Why This Approach Is Reasonable

This design is intentionally pragmatic.

### 8.1 It is explainable

Each transformation is visible:

- what was detected
- what was changed
- what metadata rule was applied
- what the synthetic generator did
- how the output was validated

That makes it much easier to defend in a judge setting than a black-box generative model.

### 8.2 It balances privacy and utility

The system does not optimize only one side.

It explicitly manages:

- fidelity
- privacy
- missingness
- outliers
- rare cases
- schema retention

That is much closer to how a real hospital team would think about release readiness.

### 8.3 It is controllable

Users can adjust:

- field inclusion
- field actions
- sensitivity handling
- locked distributions
- correlation preservation
- rare case retention
- noise level
- missingness behavior
- outlier strategy

This is important because healthcare datasets are heterogeneous and one-size-fits-all synthetic generation is rarely appropriate.

### 8.4 It is honest about limitations

The app uses heuristics and transparent rules rather than claiming mathematically perfect privacy or perfect synthetic realism.

That honesty is a strength in a hospital context.

---

## 9. Honest Limitations

For judge discussions, it is important to be clear about what the system does **not** claim.

### Not formal differential privacy

The privacy score is heuristic. It checks overlap, identifier reuse, and privacy posture, but it is not a DP guarantee.

### Not a full generative model of all dependencies

Correlation preservation uses anchor blending, not a fully learned probabilistic dependency graph.

### Not a substitute for governance

The app supports review and release controls, but governance decisions still need human approval.

### Not for clinical decision support

This synthetic output is for analytics, testing, training, and prototyping, not for patient-specific treatment decisions.

Being explicit about these limitations increases credibility.

---

## 10. Suggested Judge Explanation

Here is a concise way to explain the system:

> We do not transform raw hospital data directly into synthetic output in one black-box step.  
> First, we profile the schema and detect hygiene and sensitivity issues.  
> Then we convert the raw dataset into an explicit metadata package that says how each field should be handled.  
> After approval, we generate synthetic data using controllable rules for noise, missingness, rare-case retention, and correlation preservation.  
> Finally, we validate the result using field-level fidelity scoring, privacy overlap checks, schema coverage, and downstream utility signals.  
> The reason this is reasonable is that it is transparent, adjustable, and measurable, which is exactly what hospital teams need when balancing privacy protection with analytic usefulness.

---

## 11. Short Version: Why This Demo Is Credible

- It makes privacy controls explicit, field by field
- It preserves useful analytical structure instead of only masking values
- It uses measurable validation rather than subjective claims
- It allows human review before release
- It is simple enough to explain clearly, but strong enough to demonstrate real operational thinking

---

## 12. One-Sentence Summary

This system is credible because it treats synthetic healthcare data generation as a governed transformation pipeline, not a black-box model, and it makes every important tradeoff between privacy, fidelity, and usability visible and measurable.
