"""
Country code mapping for the geo choropleth drill-down.

Plotly's built-in choropleth needs ISO-3166 alpha-3 codes (locationmode
"ISO-3") — our data uses alpha-2 codes per the dataset glossary. Two things
can go wrong converting between them, and this module exists to handle both
without silently dropping data:

1. The glossary's own sample list includes "UK", which is NOT a valid
   ISO-3166 alpha-2 code — the real one is "GB". "UK" is aliased before
   lookup.
2. The glossary's country list is explicitly marked "e.g." — not
   exhaustive. Any code this dataset uses that isn't a real ISO alpha-2
   (or isn't in ALIASES) fails to map. Callers get both the mapped rows AND
   the list of anything that didn't map, so the UI can say so rather than
   quietly rendering an incomplete map.
"""

import pycountry

# Known non-standard codes used by this specific dataset.
ALIASES = {
    "UK": "GB",
}


def to_alpha3(code: str) -> str | None:
    """alpha-2 (or aliased) country code -> ISO alpha-3, or None if unrecognized."""
    if not code:
        return None
    normalized = ALIASES.get(code.upper(), code.upper())
    country = pycountry.countries.get(alpha_2=normalized)
    return country.alpha_3 if country else None


def to_name(code: str) -> str | None:
    """alpha-2 (or aliased) country code -> full country name, or None."""
    if not code:
        return None
    normalized = ALIASES.get(code.upper(), code.upper())
    country = pycountry.countries.get(alpha_2=normalized)
    return country.name if country else None


def map_dataframe(df, code_col: str = "dimension_value"):
    """Add `alpha3` and `country_name` columns to a drill-down DataFrame.

    Returns (mapped_df, unmapped_codes) — rows whose code_col didn't resolve
    are DROPPED from mapped_df (they can't be placed on a choropleth) but
    their original codes are returned so the caller can disclose them rather
    than silently rendering an incomplete-looking map with no explanation.
    """
    out = df.copy()
    out["alpha3"] = out[code_col].map(to_alpha3)
    out["country_name"] = out[code_col].map(to_name)
    unmapped = sorted(out.loc[out["alpha3"].isna(), code_col].unique().tolist())
    mapped = out.dropna(subset=["alpha3"])
    return mapped, unmapped