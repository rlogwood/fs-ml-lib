import pandas as pd

try:
    # When imported as part of a package
    from . import text_util as tu
except ImportError:
    # When run as a standalone script
    import text_util as tu


def inspect_variable(var, var_name="variable"):
    print(f"=== {var_name} ===")
    print(f"Class: {var.__class__}")
    print(f"Class Name: {var.__class__.__name__}")
    print(f"Type: {type(var)}")
    print(f"Type name: {type(var).__name__}")
    print(f"String representation: {str(var)[:100]}...")
    if hasattr(var, 'shape'):
        print(f"Shape: {var.shape}")
    print()


def show_breakdown(df, col1, col2, from_val: float = None, to_val: float = None):
    def print_summary(title):
        tu.print_sub_heading(title)
        # summary = df.groupby(col1)[col2].value_counts().unstack(fill_value=0)

        if from_val is not None:
            if to_val is not None:
                # print(summary.loc[from_val:to_val,:])
                print(tu.bold_text(f'Breakdown between {from_val} and {to_val}'))
                print(summary.loc[(summary.index > from_val) & (summary.index <= to_val)])
                print(tu.italic_text(tu.bold_text('Total')))
                print(summary.loc[(summary.index > from_val) & (summary.index <= to_val)].sum())
            else:
                print(tu.bold_text(f'Breakdown from {from_val}'))
                print(summary.loc[(summary.index > from_val)])
                print(tu.italic_text(tu.bold_text('Total')))
                print(summary.loc[(summary.index > from_val)].sum())
        else:
            print(summary)

    tu.print_heading(f'summary of {col1} vs {col2}')
    summary = pd.crosstab(df[col1], df[col2])
    print_summary('crosstab')

    # summary = df.groupby(col1)[col2].value_counts().unstack(fill_value=0)
    # print_summary('group_by')
    #
    # summary = df.pivot_table(index=col1, columns=col2, aggfunc='size', fill_value=0)
    # print_summary('pivot_table')

