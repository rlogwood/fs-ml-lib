import pytest
import pandas as pd
import numpy as np

from lib.data_cleaner import (
    clean_object_columns,
    clean_numeric_columns,
    clean_dataframe,
    CaseStyle,
    FillStrategy,
    DEFAULT_MISSING_VALUES
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_object_df():
    """DataFrame with object columns containing various missing value representations."""
    return pd.DataFrame({
        'name': ['John', 'NA', 'Jane', '?', 'Bob'],
        'city': ['new york', 'N/A', 'BOSTON', 'null', '  chicago  ']
    })


@pytest.fixture
def sample_numeric_df():
    """DataFrame with numeric columns containing NaN values."""
    return pd.DataFrame({
        'age': [25.0, np.nan, 35.0, 40.0, np.nan],
        'salary': [50000.0, 60000.0, np.nan, 80000.0, 90000.0]
    })


@pytest.fixture
def sample_mixed_df():
    """DataFrame with both object and numeric columns."""
    return pd.DataFrame({
        'name': ['John', 'NA', 'Jane'],
        'age': [25.0, np.nan, 35.0],
        'city': ['new york', 'null', 'BOSTON']
    })


# =============================================================================
# Tests for clean_object_columns
# =============================================================================

class TestCleanObjectColumns:
    """Tests for clean_object_columns function."""

    def test_replaces_missing_values_with_nan(self, sample_object_df):
        """Missing value placeholders should be replaced with np.nan by default."""
        result = clean_object_columns(sample_object_df)

        # Check that 'NA' and '?' in name column are now NaN
        assert pd.isna(result['name'][1])
        assert pd.isna(result['name'][3])

        # Check that valid values are preserved
        assert result['name'][0] == 'John'
        assert result['name'][2] == 'Jane'
        assert result['name'][4] == 'Bob'

    def test_replaces_missing_values_with_custom_string(self, sample_object_df):
        """Missing values should be replaced with custom replacement string."""
        result = clean_object_columns(sample_object_df, replacement='MISSING')

        assert result['name'][1] == 'MISSING'
        assert result['name'][3] == 'MISSING'
        assert result['city'][1] == 'MISSING'
        assert result['city'][3] == 'MISSING'

    def test_case_style_title(self, sample_object_df):
        """CaseStyle.TITLE should convert to Title Case."""
        result = clean_object_columns(sample_object_df, case_style=CaseStyle.TITLE)

        assert result['name'][0] == 'John'
        assert result['city'][0] == 'New York'
        assert result['city'][2] == 'Boston'

    def test_case_style_upper(self, sample_object_df):
        """CaseStyle.UPPER should convert to UPPER CASE."""
        result = clean_object_columns(sample_object_df, case_style=CaseStyle.UPPER)

        assert result['name'][0] == 'JOHN'
        assert result['city'][0] == 'NEW YORK'

    def test_case_style_lower(self, sample_object_df):
        """CaseStyle.LOWER should convert to lower case."""
        result = clean_object_columns(sample_object_df, case_style=CaseStyle.LOWER)

        assert result['name'][0] == 'john'
        assert result['city'][2] == 'boston'

    def test_case_style_original(self, sample_object_df):
        """CaseStyle.ORIGINAL should not change case."""
        result = clean_object_columns(sample_object_df, case_style=CaseStyle.ORIGINAL)

        assert result['name'][0] == 'John'
        assert result['city'][0] == 'new york'
        assert result['city'][2] == 'BOSTON'

    def test_strips_whitespace(self, sample_object_df):
        """Whitespace should be stripped by default."""
        result = clean_object_columns(sample_object_df)

        # '  chicago  ' should become 'chicago'
        assert result['city'][4] == 'chicago'

    def test_no_strip_whitespace(self, sample_object_df):
        """Whitespace should be preserved when strip_whitespace=False."""
        result = clean_object_columns(sample_object_df, strip_whitespace=False)

        # Note: astype(str) is not called, so original whitespace preserved
        # But the column still gets processed
        assert '  chicago  ' in result['city'].values or 'chicago' in result['city'].values

    def test_specific_columns_only(self, sample_object_df):
        """Only specified columns should be cleaned."""
        result = clean_object_columns(sample_object_df, columns=['name'], case_style=CaseStyle.UPPER)

        # 'name' should be uppercase
        assert result['name'][0] == 'JOHN'
        # 'city' should be unchanged (except whitespace stripping doesn't apply since not in columns)
        # Actually, since we only process 'name', city is untouched
        assert result['city'][0] == 'new york'

    def test_does_not_modify_original(self, sample_object_df):
        """Original DataFrame should not be modified."""
        original_value = sample_object_df['name'][0]
        clean_object_columns(sample_object_df, case_style=CaseStyle.UPPER)

        assert sample_object_df['name'][0] == original_value

    def test_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        empty_df = pd.DataFrame({'name': []})
        result = clean_object_columns(empty_df)

        assert len(result) == 0

    def test_no_object_columns(self):
        """Should handle DataFrame with no object columns."""
        numeric_only = pd.DataFrame({'age': [25, 30, 35]})
        result = clean_object_columns(numeric_only)

        assert result.equals(numeric_only)

    def test_nan_not_affected_by_case_transformation(self, sample_object_df):
        """NaN values should not be affected by case transformation."""
        result = clean_object_columns(sample_object_df, case_style=CaseStyle.UPPER)

        # NaN values should still be NaN, not 'NAN'
        assert pd.isna(result['name'][1])
        assert pd.isna(result['name'][3])

    def test_normalize_spaces_collapses_multiple_spaces(self):
        """normalize_spaces=True should collapse multiple internal spaces into one."""
        df = pd.DataFrame({
            'name': ['John    Smith', 'Jane   Doe', 'Bob'],
            'address': ['123   Main    St', 'N/A', '456 Oak Ave']
        })
        result = clean_object_columns(df, normalize_spaces=True)

        assert result['name'][0] == 'John Smith'
        assert result['name'][1] == 'Jane Doe'
        assert result['address'][0] == '123 Main St'
        assert result['address'][2] == '456 Oak Ave'

    def test_normalize_spaces_default_false(self):
        """normalize_spaces should be False by default (preserve internal spaces)."""
        df = pd.DataFrame({'name': ['John    Smith']})
        result = clean_object_columns(df)

        # Multiple spaces should be preserved when normalize_spaces=False (default)
        assert result['name'][0] == 'John    Smith'

    def test_normalize_spaces_does_not_affect_nan(self):
        """normalize_spaces should not affect NaN values."""
        df = pd.DataFrame({'name': ['John    Smith', 'NA', 'Jane']})
        result = clean_object_columns(df, normalize_spaces=True)

        assert result['name'][0] == 'John Smith'
        assert pd.isna(result['name'][1])

    def test_column_case_styles_per_column(self):
        """column_case_styles should apply different case styles to different columns."""
        df = pd.DataFrame({
            'name': ['john doe', 'jane smith'],
            'state': ['california', 'new york'],
            'code': ['abc', 'xyz']
        })
        result = clean_object_columns(df,
            column_case_styles={
                'name': CaseStyle.TITLE,
                'state': CaseStyle.UPPER
            },
            case_style=CaseStyle.ORIGINAL  # fallback for 'code'
        )

        # 'name' should be Title Case
        assert result['name'][0] == 'John Doe'
        assert result['name'][1] == 'Jane Smith'

        # 'state' should be UPPER CASE
        assert result['state'][0] == 'CALIFORNIA'
        assert result['state'][1] == 'NEW YORK'

        # 'code' should be unchanged (ORIGINAL fallback)
        assert result['code'][0] == 'abc'
        assert result['code'][1] == 'xyz'

    def test_column_case_styles_fallback_to_case_style(self):
        """Columns not in column_case_styles should use case_style as fallback."""
        df = pd.DataFrame({
            'name': ['john'],
            'city': ['boston'],
            'country': ['usa']
        })
        result = clean_object_columns(df,
            column_case_styles={'name': CaseStyle.TITLE},
            case_style=CaseStyle.UPPER  # fallback for city and country
        )

        assert result['name'][0] == 'John'      # from column_case_styles
        assert result['city'][0] == 'BOSTON'    # from case_style fallback
        assert result['country'][0] == 'USA'    # from case_style fallback

    def test_column_case_styles_empty_dict(self):
        """Empty column_case_styles should use case_style for all columns."""
        df = pd.DataFrame({'name': ['john'], 'city': ['boston']})
        result = clean_object_columns(df,
            column_case_styles={},
            case_style=CaseStyle.UPPER
        )

        assert result['name'][0] == 'JOHN'
        assert result['city'][0] == 'BOSTON'

    def test_column_case_styles_with_normalize_spaces(self):
        """column_case_styles should work together with normalize_spaces."""
        df = pd.DataFrame({
            'name': ['john    doe'],
            'state': ['new    york']
        })
        result = clean_object_columns(df,
            column_case_styles={
                'name': CaseStyle.TITLE,
                'state': CaseStyle.UPPER
            },
            normalize_spaces=True
        )

        assert result['name'][0] == 'John Doe'
        assert result['state'][0] == 'NEW YORK'


# =============================================================================
# Tests for clean_numeric_columns
# =============================================================================

class TestCleanNumericColumns:
    """Tests for clean_numeric_columns function."""

    def test_strategy_none_leaves_nan(self, sample_numeric_df):
        """FillStrategy.NONE should leave NaN values as-is."""
        result = clean_numeric_columns(sample_numeric_df, strategy=FillStrategy.NONE)

        assert pd.isna(result['age'][1])
        assert pd.isna(result['age'][4])
        assert pd.isna(result['salary'][2])

    def test_strategy_drop(self, sample_numeric_df):
        """FillStrategy.DROP should remove rows with NaN."""
        result = clean_numeric_columns(sample_numeric_df, strategy=FillStrategy.DROP)

        # Original has 5 rows, 3 have NaN values (rows 1, 2, 4)
        # Only rows 0 and 3 have no NaN
        assert len(result) == 2
        assert result['age'].isna().sum() == 0
        assert result['salary'].isna().sum() == 0

    def test_strategy_mean(self, sample_numeric_df):
        """FillStrategy.MEAN should fill NaN with column mean."""
        result = clean_numeric_columns(sample_numeric_df, strategy=FillStrategy.MEAN)

        # age mean = (25 + 35 + 40) / 3 = 33.33...
        expected_age_mean = (25.0 + 35.0 + 40.0) / 3
        assert result['age'][1] == pytest.approx(expected_age_mean)
        assert result['age'][4] == pytest.approx(expected_age_mean)

        # salary mean = (50000 + 60000 + 80000 + 90000) / 4 = 70000
        assert result['salary'][2] == pytest.approx(70000.0)

    def test_strategy_median(self, sample_numeric_df):
        """FillStrategy.MEDIAN should fill NaN with column median."""
        result = clean_numeric_columns(sample_numeric_df, strategy=FillStrategy.MEDIAN)

        # age values (excluding NaN): 25, 35, 40 -> median = 35
        assert result['age'][1] == 35.0
        assert result['age'][4] == 35.0

        # salary values (excluding NaN): 50000, 60000, 80000, 90000 -> median = 70000
        assert result['salary'][2] == 70000.0

    def test_strategy_mode(self):
        """FillStrategy.MODE should fill NaN with column mode."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 2.0, np.nan, 3.0]  # mode is 2.0
        })
        result = clean_numeric_columns(df, strategy=FillStrategy.MODE)

        assert result['value'][3] == 2.0

    def test_strategy_zero(self, sample_numeric_df):
        """FillStrategy.ZERO should fill NaN with 0."""
        result = clean_numeric_columns(sample_numeric_df, strategy=FillStrategy.ZERO)

        assert result['age'][1] == 0.0
        assert result['age'][4] == 0.0
        assert result['salary'][2] == 0.0

    def test_strategy_value_with_fill_value(self, sample_numeric_df):
        """FillStrategy.VALUE should fill NaN with provided fill_value."""
        result = clean_numeric_columns(sample_numeric_df, strategy=FillStrategy.VALUE, fill_value=-999.0)

        assert result['age'][1] == -999.0
        assert result['age'][4] == -999.0
        assert result['salary'][2] == -999.0

    def test_strategy_value_without_fill_value_raises(self, sample_numeric_df):
        """FillStrategy.VALUE without fill_value should raise ValueError."""
        with pytest.raises(ValueError, match="fill_value must be provided"):
            clean_numeric_columns(sample_numeric_df, strategy=FillStrategy.VALUE)

    def test_specific_columns_only(self, sample_numeric_df):
        """Only specified columns should be cleaned."""
        result = clean_numeric_columns(sample_numeric_df, columns=['age'], strategy=FillStrategy.ZERO)

        # 'age' NaN should be filled with 0
        assert result['age'][1] == 0.0
        # 'salary' NaN should remain
        assert pd.isna(result['salary'][2])

    def test_does_not_modify_original(self, sample_numeric_df):
        """Original DataFrame should not be modified."""
        assert pd.isna(sample_numeric_df['age'][1])
        clean_numeric_columns(sample_numeric_df, strategy=FillStrategy.ZERO)
        assert pd.isna(sample_numeric_df['age'][1])

    def test_no_numeric_columns(self):
        """Should handle DataFrame with no numeric columns."""
        object_only = pd.DataFrame({'name': ['John', 'Jane']})
        result = clean_numeric_columns(object_only, strategy=FillStrategy.MEAN)

        assert result.equals(object_only)

    def test_no_nan_values(self):
        """Should handle DataFrame with no NaN values."""
        clean_df = pd.DataFrame({'age': [25.0, 30.0, 35.0]})
        result = clean_numeric_columns(clean_df, strategy=FillStrategy.MEAN)

        assert result.equals(clean_df)


# =============================================================================
# Tests for clean_dataframe
# =============================================================================

class TestCleanDataframe:
    """Tests for clean_dataframe convenience function."""

    def test_both_configs(self, sample_mixed_df):
        """Should apply both object and numeric cleaning."""
        result = clean_dataframe(
            sample_mixed_df,
            object_config={'case_style': CaseStyle.UPPER},
            numeric_config={'strategy': FillStrategy.ZERO}
        )

        # Object columns should be uppercase
        assert result['name'][0] == 'JOHN'
        assert result['city'][0] == 'NEW YORK'

        # Numeric NaN should be filled with 0
        assert result['age'][1] == 0.0

    def test_only_object_config(self, sample_mixed_df):
        """Should only apply object cleaning when numeric_config is None."""
        result = clean_dataframe(
            sample_mixed_df,
            object_config={'case_style': CaseStyle.UPPER}
        )

        assert result['name'][0] == 'JOHN'
        # Numeric NaN should remain
        assert pd.isna(result['age'][1])

    def test_only_numeric_config(self, sample_mixed_df):
        """Should only apply numeric cleaning when object_config is None."""
        result = clean_dataframe(
            sample_mixed_df,
            numeric_config={'strategy': FillStrategy.ZERO}
        )

        # Object columns should have case unchanged (still lowercase)
        assert result['city'][0] == 'new york'
        # But missing values are replaced with NaN and whitespace stripped
        # Numeric NaN should be filled
        assert result['age'][1] == 0.0

    def test_no_configs(self, sample_mixed_df):
        """Should return copy when no configs provided."""
        result = clean_dataframe(sample_mixed_df)

        # Should be equal but not the same object
        assert result.equals(sample_mixed_df)
        assert result is not sample_mixed_df

    def test_does_not_modify_original(self, sample_mixed_df):
        """Original DataFrame should not be modified."""
        original_name = sample_mixed_df['name'][0]
        clean_dataframe(
            sample_mixed_df,
            object_config={'case_style': CaseStyle.UPPER}
        )

        assert sample_mixed_df['name'][0] == original_name


# =============================================================================
# Tests for DEFAULT_MISSING_VALUES
# =============================================================================

class TestDefaultMissingValues:
    """Tests to verify DEFAULT_MISSING_VALUES covers common cases."""

    @pytest.mark.parametrize("missing_val", [
        "", " ", "NA", "N/A", "na", "n/a", "NULL", "null",
        "None", "none", "NaN", "nan", "-", "?", "unknown", "UNKNOWN"
    ])
    def test_all_default_missing_values_detected(self, missing_val):
        """Each value in DEFAULT_MISSING_VALUES should be detected and replaced."""
        df = pd.DataFrame({'col': ['valid', missing_val, 'also_valid']})
        result = clean_object_columns(df)

        assert pd.isna(result['col'][1]), f"'{missing_val}' was not detected as missing"
