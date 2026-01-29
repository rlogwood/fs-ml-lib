# TODO: Notes to review and improve

- What it guarantees: no pandas-native missing markers (NaN/NaT/None) are present in any column.
- What it doesn’t catch:
   - “Missing” encoded as strings like "", " ", "NA", "N/A", "-", "null", "None", "?".
   - Columns read as object because of mixed types where missing are hidden as text.
   - Out-of-range placeholders (e.g., -1, 9999) that semantically mean missing.
   - Datetime parsing failures if the column is still string.

 Recommendations:
 - Inspect object columns for suspicious values:
   - sales.select_dtypes(include="object").agg(lambda s: s.str.strip().isin(["", "NA","N/A","-","null","None","?"]).sum())
 - Normalize such tokens to NaN, then re-run isna().sum().
 - For numerics that came in as strings, coerce:
   - sales[col] = pd.to_numeric(sales[col], errors="coerce")
 - For dates:
   - sales[col] = pd.to_datetime(sales[col], errors="coerce")
 - Check whitespace-only cells:
   - sales.select_dtypes("object").apply(lambda s: s.str.strip().eq("").sum())

 If all of the above show zero after cleaning, you can confidently state there’s no missing data.
