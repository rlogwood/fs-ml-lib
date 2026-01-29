from pandas import describe_option

try:
    # When imported as part of a package
    from . import text_util as tu
except ImportError:
    # When run as a standalone script
    import text_util as tu

# Check column names and sample values first
def print_col_info(df):
    tu.styled_display("Column info:")
    print(f"{tu.bold_text('Columns:')} {df.columns.tolist()}")
    print(f"{tu.bold_text('Shape:')} {df.shape}")
    print(f"\n{tu.bold_text('Dtypes:')}\n{'='*7}\n{df.dtypes}")

    print(f"\n{tu.bold_text('Sample values for object columns:')}")
    print("="*70)
    for col in df.select_dtypes("object").columns:
        print(f"{col}: {df[col].head(3).tolist()}")


def print_missing_values_counts(df):
    # Standard NA Check
    tu.styled_display("Standard NA Check:")
    print(df.isna().sum())

    # Check object columns for bad values
    #obj_cols = df.select_dtypes(include="object").agg(
    #    lambda s: s.str.strip().isin(["", "NA", "N/A", "-", "null", "None", "?"]).sum())

    obj_cols = df.select_dtypes(include="object").agg(
        lambda s: s.str.strip().isin(im.DEFAULT_MISSING_VALUES).sum())
    tu.styled_display("Bad object Values Check:")
    print(obj_cols)


def show_group_by_analysis(df, group_by_col, agg_col):
    tu.styled_display(f"Group by analysis for {group_by_col}:")
    analysis = df.groupby(group_by_col)[agg_col].agg(['mean', 'count', 'sum'])
    analysis.loc['Total'] = analysis.sum()
    print("")
    description = f"{agg_col} Analysis"
    print(tu.print_sub_heading(description))
    print(analysis)


def show_group_by_analysis_multi(df, group_by_col, agg_cols, description, sort_by = None):
    analysis = df.groupby(group_by_col).agg(agg_cols).round(2)

    # Debugging Stuff
    # try:
    #     print(f"sort_by: {sort_by}")
    #     print(f"sort_by[0]: {sort_by[0]}")
    #     print(f"sort_by[1]: {sort_by[1]}")
    #
    #     print(f"agg_cols: {agg_cols}")
    #     print(f"agg_cols[0]: {agg_cols[0]}, agg_cols[0][0]: {agg_cols[0][0]}")
    # except Exception as e:
    #     print(f"Exception: {e}")

    if sort_by:
        try:
            analysis = analysis.sort_values((sort_by[0], sort_by[1]), ascending=False)
        except Exception as e:
            print(f"Sorting Exception: {e}")

    analysis.loc['Total'] = analysis.sum()
    analysis = analysis.map(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
    print("")
    print(tu.print_sub_heading(description))
    print(analysis)


def add_commas(data):
    return data.map(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)


def show_unique_values_for_object_columns(df, max_unique=50):
    """
    Display unique values for all object (string) columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze
    max_unique (int): Maximum number of unique values to display per column

    Returns:
    dict: Dictionary with column names as keys and unique values as values
    """
    object_cols = df.select_dtypes(include=['object']).columns

    if len(object_cols) == 0:
        print("No object columns found in the DataFrame.")
        return {}

    unique_values_dict = {}

    tu.print_heading(f"Unique values for object columns (up to {max_unique}):")

    for col in object_cols:
        unique_vals = df[col].unique()
        unique_count = len(unique_vals)

        print(f"\n📊 Column: {col}")
        print(f"   Total unique values: {unique_count}")

        if unique_count <= max_unique:
            print(f"   Values: {list(unique_vals)}")
        else:
            print(f"   First {max_unique} values: {list(unique_vals[:max_unique])}")
            print(f"   ... and {unique_count - max_unique} more")

        # Also show value counts
        value_counts = df[col].value_counts()

        print(f"   Most frequent: {value_counts.index[0]} ({value_counts.iloc[0]} times)")

        unique_values_dict[col] = unique_vals

    return unique_values_dict

