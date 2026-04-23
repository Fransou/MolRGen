import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler


class PandasTableFormatter:
    def __init__(
        self,
        n_decimals: int = 3,
        aggregation_methods: List[Any] = ["mean", "std"],
        main_subset: int = 0,
        total_col_name: str = "AVG.",
        hide_agg_labels: bool = True,
        already_rotated: bool = False,
        global_agg: bool = True,
        color_mapping: Union[Callable[[float], str], None] = None,
        groupby_col: str | None = None,
        color_mapping_key_idx: int | None = None,
        merge_aggs: bool = False,
    ):
        """
        PandasTableFormatter is a class that formats a Pandas DataFrame into a LaTeX
        table with custom aggregation methods and styles.

        The 'main_subset' parameter is used to specify which aggregation methods should be used
        to compare the different rows in the table (often the mean). These aggregagtion results
        will be the one highlighted in the latex table.

        If main_subset is specified, a column ($NAME, agg$) will be created with the aggregation method

        :param color_mapping: Optional callable that takes a float value and returns a color string (e.g., '#FF5733').
                            This function will be applied to each cell to assign background colors based on values.
        :param color_mapping_key_idx: Optional index specifying which aggregation method's value should be used
                            to determine the color for all related cells (e.g., use mean's value to color both mean and std).
                            If None, color_mapping is applied to each cell independently.
        :param merge_aggs: If True, merges all aggregated values into a single cell (e.g., "1.5±0.5" instead of separate columns).
                          The formatting is determined by special_format_agg for non-main aggregations.
        """
        self.n_decimals = n_decimals
        self.aggregation_methods = aggregation_methods
        self.hide_agg_labels = hide_agg_labels
        self.already_rotated = already_rotated
        self.global_agg = global_agg
        self.color_mapping = color_mapping
        self.groupby_col = groupby_col
        self.color_mapping_key_idx = color_mapping_key_idx
        self.merge_aggs = merge_aggs

        for agg in self.aggregation_methods:
            if not isinstance(agg, str) and not callable(agg):
                raise ValueError(
                    "Aggregation methods must be either a string or a callable function"
                )
            elif callable(agg):
                if not hasattr(agg, "__name__"):
                    raise ValueError(
                        "Aggregation function must have a __name__ attribute"
                    )
        assert 0 <= main_subset < len(aggregation_methods), (
            "main_subset must be an integer between 0 and the number of aggregation methods"
        )

        if color_mapping_key_idx is not None:
            assert 0 <= color_mapping_key_idx < len(aggregation_methods), (
                "color_mapping_key_idx must be an integer between 0 and the number of aggregation methods"
            )

        self.main_subset = main_subset
        if isinstance(aggregation_methods[main_subset], str):
            self.main_agg = aggregation_methods[main_subset]
        elif hasattr(aggregation_methods[main_subset], "__main__"):
            self.main_agg = aggregation_methods[main_subset].__name__
        self.total_col_name = total_col_name

    def _find_k_th_fn(
        self,
        s: np.ndarray | pd.Series,
        fn: Callable[[np.ndarray | pd.Series], float],
        k: int,
        props: str = "",
    ) -> List[str]:
        """
        Highlight the top k values in a dataframe using the prop string.

        :param s: The input array or series to be processed.
        :param fn: The function to be applied to the input.
        :param k: The number of top values to highlight.
        :param props: The properties to be applied to the highlighted values.

        :return: A list of strings with the highlighted values.
        """
        ps_list = [np.where(s == fn(s), True, False)]
        for i in range(1, k):
            previous_ps = np.concatenate(ps_list).reshape(i, -1).any(axis=0)
            ps_list.append(np.where(s == fn(s[~previous_ps]), True, False))
        return [props if p else "" for p in ps_list[-1]]

    def _find_k_th_fn_for_merge(
        self,
        df: pd.DataFrame,
        fn: Callable[[np.ndarray | pd.Series], float],
        k: int,
        props: str = "",
        main_agg: str = "mean",
    ) -> List[tuple]:
        """
        Find top k values for highlighting and return their indices and columns.
        Used for merge_aggs to preserve highlighting after merging.

        :param df: The aggregated dataframe.
        :param fn: The highlight function (e.g., np.nanmax).
        :param k: The number of top values to highlight.
        :param props: The CSS properties to apply.
        :param main_agg: The main aggregation method name.
        :return: List of (row_index, column) tuples that should be highlighted.
        """
        highlighted = []

        # Get columns that match the main aggregation
        main_agg_cols = [c for c in df.columns if c[-1] == main_agg]

        for col in main_agg_cols:
            s = df[col]
            ps_list = [np.where(s == fn(s), True, False)]
            for i in range(1, k):
                previous_ps = np.concatenate(ps_list).reshape(i, -1).any(axis=0)
                ps_list.append(np.where(s == fn(s[~previous_ps]), True, False))

            # Get indices where this column should be highlighted
            mask = ps_list[-1]
            for row_idx in df.index:
                if mask[df.index.get_loc(row_idx)]:
                    highlighted.append((row_idx, col))

        return highlighted
        """
        Apply color mapping to a series of values based on the color_mapping function.

        :param s: The input array or series to be processed.
        :return: A list of CSS background-color properties for each value.
        """
        if self.color_mapping is None:
            return [""] * len(s)

        colors = []
        for val in s:
            if pd.isna(val):
                colors.append("")
            else:
                try:
                    color = self.color_mapping(float(val))
                    colors.append(f"background-color: {color};")
                except (ValueError, TypeError):
                    colors.append("")
        return colors

    def _apply_color_mapping_with_key(
        self, df: pd.DataFrame, color_mapping_key_idx: int
    ) -> pd.DataFrame:
        """
        Apply color mapping based on a specific aggregation method's value to all related cells.
        Groups cells by their base column (all levels except the last 'agg' level) and applies
        color based on the value from the specified aggregation index.

        :param df: The aggregated dataframe with MultiIndex columns.
        :param color_mapping_key_idx: Index of the aggregation method to use for color determination.
        :return: A dataframe of CSS background-color properties.
        """
        if self.color_mapping is None:
            return pd.DataFrame("", index=df.index, columns=df.columns)

        # Get the aggregation method name at the specified index
        agg_method = self.aggregation_methods[color_mapping_key_idx]
        key_agg_name = (
            agg_method if isinstance(agg_method, str) else agg_method.__name__
        )

        # Create output dataframe with same shape
        colors_df = pd.DataFrame("", index=df.index, columns=df.columns)

        # Group columns by their base (all levels except the last 'agg' level)
        grouped_cols: dict[str, list] = {}
        for col in df.columns:
            base_col = col[:-1]  # All levels except the last (agg) level
            if base_col not in grouped_cols:
                grouped_cols[base_col] = []
            grouped_cols[base_col].append(col)

        # For each group, get the color from the key aggregation method and apply to all
        for base_col, cols_in_group in grouped_cols.items():
            # Find the column with the key aggregation method
            key_col = None
            for col in cols_in_group:
                if col[-1] == key_agg_name:
                    key_col = col
                    break

            if key_col is not None:
                # Apply color mapping based on key column values to all columns in the group
                for row_idx in df.index:
                    val = df.loc[row_idx, key_col]
                    if pd.isna(val):
                        color = ""
                    else:
                        try:
                            color = self.color_mapping(float(val))
                            color = f"background-color: {color};"
                        except (ValueError, TypeError):
                            color = ""

                    # Apply this color to all cells in the group for this row
                    for col in cols_in_group:
                        colors_df.loc[row_idx, col] = color

        return colors_df

    def _merge_aggregations(
        self,
        df_agg: pd.DataFrame,
        special_format_agg: Dict[str, Callable[[str], str]],
    ) -> pd.DataFrame:
        """
        Merge all aggregation methods for each base column into a single concatenated value.

        :param df_agg: The aggregated dataframe with MultiIndex columns ([col_levels..., agg]).
        :param special_format_agg: Dictionary mapping aggregation names to formatting functions.
        :return: A dataframe with merged aggregations (agg level removed from MultiIndex).
        """
        # Create a new dataframe with merged columns
        merged_data = {}

        # Group columns by their base (all levels except the last 'agg' level)
        grouped_cols: dict[str, list] = {}
        for col in df_agg.columns:
            base_col = col[:-1]  # All levels except the last (agg) level
            if base_col not in grouped_cols:
                grouped_cols[base_col] = []
            grouped_cols[base_col].append(col)

        # For each base column, merge all aggregations
        for base_col, cols_in_group in grouped_cols.items():
            merged_col_values = []

            for row_idx in df_agg.index:
                merged_values = []

                for col in cols_in_group:
                    val = df_agg.loc[row_idx, col]
                    agg_name = col[-1]

                    if pd.isna(val):
                        val_str = "NaN"
                    else:
                        val_str = str(np.round(val, self.n_decimals))

                    # Apply special formatting if available
                    if agg_name in special_format_agg:
                        val_str = special_format_agg[agg_name](val_str)

                    merged_values.append(val_str)

                # Concatenate all values for this row
                merged_col_values.append(" ".join(merged_values))

            # Add the merged column to the output
            merged_data[base_col] = merged_col_values

        # Create the new dataframe with merged columns
        merged_df = pd.DataFrame(merged_data, index=df_agg.index)

        # Convert column names from tuples to regular index
        merged_df.columns = pd.Index(
            [col for col in merged_df.columns],
            name=None,
        )

        return merged_df

    def _aggregate_results_and_pivot(
        self,
        df_base: pd.DataFrame,
        rows: str | List[str],
        cols: str | List[str],
        values: str,
    ) -> pd.DataFrame:
        """
        Aggregates the given dataframe by computing the mean and standard deviation for a specified
        metric, organizing the results in a structured format.

        Example with no multicols:
            df_base = pd.DataFrame(
                {
                    "name": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
                    "category": ["A", "A", "A", "B", "B", "B", "A", "A", "A", "B", "B", "B"],
                    "value": [0,1,2,-1,0,1,-1,1,3,-10,0,10],
                }
            )
            self._aggregate_results_with_std(df_base)

            >> Output:
                 avg mean   avg std   A mean    A std    B mean      B std
            a     0.5        0.7        1         1        0          0.5
            b     0.5        0.7        1         2        0          10

        Example with multicols {"A": ["category1", "A"], "B": ["category1", "B"], "C": ["category2", "C"]}:
            df_base = pd.DataFrame(
                {
                    "name": ["a", "a", "a", "a", "b", "b", "b", "b"],
                    "category": ["A", "A", "B", "B", "A", "A", "B", "B"],
                    "value": [0,1,2,2,0,1,2,2],
                }
            )
            self._aggregate_results_with_std(df_base)

            >> Output:
                 avg    avg   A    A    B    B
                mean   std  mean  std  mean std
            a    0.5   0.5   1    1    2    0
            b    0.5   0.5   1    1    2    0


        :param df_base: The base dataframe containing the data to be aggregated.
        :param rows: The column(s) to be used as rows in the resulting dataframe.
        :param cols: The column(s) to be used as columns in the resulting dataframe.
        :param values: The column(s) to be used as values in the resulting dataframe.
        :param aggregation_methods: The aggregation functions to be applied to the values.

        :return:  A formatted dataframe containing aggregated mean and standard
        deviation values.
        """
        # Join the mean and std dataframes to a new one
        dataframes_to_concatenate = []
        df_glob = []
        for i_agg_meth, agg in enumerate(self.aggregation_methods):
            if isinstance(rows, str):
                rows = [rows]
            if self.groupby_col is not None:
                rows_p = rows + [self.groupby_col]
            else:
                rows_p = rows
            df_agg = df_base.pivot_table(
                index=rows_p, columns=cols, values=values, aggfunc=agg
            )
            if self.groupby_col is not None:
                df_agg = df_agg.groupby(rows).mean()
            df_agg.columns = pd.MultiIndex.from_arrays(
                [df_agg.columns.get_level_values(i) for i in range(len(cols))]
                + [
                    pd.Index(
                        [agg if isinstance(agg, str) else agg.__name__]
                        * df_agg.shape[1],
                        name="agg",
                    )
                ],
                names=cols + ["agg"] if isinstance(cols, list) else [cols, "agg"],
            )

            if i_agg_meth == self.main_subset and self.global_agg:
                for agg_b in self.aggregation_methods:
                    df_glob_agg = df_agg.agg(agg_b, axis=1)
                    df_glob_agg = df_glob_agg.to_frame(
                        name=agg_b if isinstance(agg_b, str) else agg_b.__name__
                    )
                    df_glob_agg.columns = pd.MultiIndex.from_arrays(
                        [
                            pd.Index(
                                [" " if i < len(cols) - 1 else self.total_col_name],
                                name=cols[i],
                            )
                            for i in range(len(cols))
                        ]
                        + [
                            pd.Index(
                                [agg_b if isinstance(agg_b, str) else agg_b.__name__],
                                name="agg",
                            )
                        ],
                        names=(
                            cols + ["agg"] if isinstance(cols, list) else [cols, "agg"]
                        ),
                    )
                    df_glob.append(df_glob_agg)
            dataframes_to_concatenate.append(df_agg)
        df_agg = pd.concat(dataframes_to_concatenate, axis=1)
        if self.global_agg:
            df_glob = pd.concat(df_glob, axis=1)
            df_agg = pd.concat([df_agg, df_glob], axis=1)
        df_agg.index.name = None
        try:
            # If one of the column's key contains a number, use the float of this number
            int_pattern = re.compile(r"(\d+)")
            idx_int: List[
                tuple[int, ...]
            ] = []  # idx of all levels that have an int, for each col
            for col_names in df_agg.columns:
                int_cols = []
                for i_elem, col_name in enumerate(col_names[:-1]):
                    if isinstance(col_name, str):
                        match = int_pattern.search(col_name)
                        if match:
                            int_cols.append(i_elem)
                idx_int.append(tuple(int_cols))
            if len(list(set(idx_int))) == 1:  # all columns have the same idx with int

                def key_fn(x: Any) -> Any:
                    key = []
                    for i, col_name in enumerate(x[:-1]):
                        if i in idx_int[0]:
                            match = int_pattern.search(col_name)
                            key.append(int(match.group(1)))  # type: ignore
                        else:
                            key.append(col_name)
                    return tuple(key)
            else:

                def key_fn(x: Any) -> Any:
                    return tuple(x[:-1])

            df_agg = df_agg.reindex(
                sorted(df_agg.columns, key=key_fn),
                axis=1,
            )
        except TypeError as e:
            print(e)
            warnings.warn(
                "Warning: Unable to sort columns. Ensure that the columns are of the same type."
            )
        return df_agg

    def style(
        self,
        df: pd.DataFrame,
        rows: str | List[str],
        cols: str | List[str],
        values: str,
        highlight_fn: Callable[[np.ndarray | pd.Series], float] = np.nanmax,
        props: List[str] = ["font-weight: bold;"],
        special_format_agg: Dict[str, Callable[[str], str]] = {
            "std": lambda x: "\\tiny $\\pm$" + x
        },
        remove_col_names: bool = False,
        row_order: Any | None = None,
    ) -> Styler:
        """
        Applies the highlight method to the given dataframe and returns a styled dataframe.
        If the Dataframe is already rotated, it will be melted first, where the columns denote
        the columns to keep, and value is the name given to the columns in the dataframe once melted.


        :param df: The dataframe to be styled.
        :return: A styled dataframe with highlighted values.
        """
        k = len(props)
        if self.already_rotated:
            df = df.melt(
                id_vars=rows,
                value_vars=cols,
                var_name=values,
                value_name="value",
            )
            cols = values
            values = "value"

        if isinstance(rows, str):
            rows = [rows]
        if isinstance(cols, str):
            cols = [cols]

        df_agg = self._aggregate_results_and_pivot(
            df,
            rows=rows,
            cols=cols,
            values=values,
        )

        # Apply highlighting to the original df_agg before merging
        highlight_styles = None
        if not self.merge_aggs:
            # Highlighting will be applied later in the normal flow
            pass
        else:
            # For merge_aggs, we need to capture highlighting info before merging
            # Apply highlighting to find which cells should be highlighted
            highlighted_cells: dict[
                Any, dict[str, str]
            ] = {}  # row_idx -> {base_col: props}
            for i in range(1, len(props) + 1):
                highlight_result = self._find_k_th_fn_for_merge(
                    df_agg,
                    fn=highlight_fn,
                    k=i,
                    props=props[i - 1],
                    main_agg=self.main_agg,
                )
                for idx, col in highlight_result:
                    if idx not in highlighted_cells:
                        highlighted_cells[idx] = {}
                    base_col = col[:-1]
                    highlighted_cells[idx][base_col] = props[i - 1]
            highlight_styles = highlighted_cells

        # Compute colors before merging (if color mapping is requested)
        colors_df_before_merge = None
        if self.color_mapping is not None and self.color_mapping_key_idx is not None:
            colors_df_before_merge = self._apply_color_mapping_with_key(
                df_agg, self.color_mapping_key_idx
            )

        # Merge aggregations if requested
        if self.merge_aggs:
            df_agg = self._merge_aggregations(df_agg, special_format_agg)
            # When merging, we no longer need special formatting as it's applied during merge
            formatter = {}

            # Convert colors_df to match the merged structure
            if colors_df_before_merge is not None:
                # Group columns by their base and merge colors
                merged_colors = {}
                grouped_cols: dict[Any, list[str]] = {}

                for col in colors_df_before_merge.columns:
                    base_col = col[:-1] if isinstance(col, tuple) else col
                    if base_col not in grouped_cols:
                        grouped_cols[base_col] = []
                    grouped_cols[base_col].append(col)

                for base_col, cols_in_group in grouped_cols.items():
                    # Use the color from the main_agg column if available
                    if len(cols_in_group) > 0:
                        merged_colors[base_col] = colors_df_before_merge[
                            cols_in_group[0]
                        ]

                colors_df_before_merge = pd.DataFrame(merged_colors, index=df_agg.index)

        # Trick as there is a data leakage (pandas issue)
        def wrap_special_format_agg(fn: Callable[[str], str]) -> Callable[[float], str]:
            def wrapped(x: float) -> str:
                x_str = str(np.round(x, self.n_decimals))
                out = fn(x_str)
                return out

            return wrapped

        if not self.merge_aggs:
            formatter = {
                c: wrap_special_format_agg(special_format_agg[c[-1]])
                for c in df_agg.columns
                if c[-1] in special_format_agg
            }
        if remove_col_names and len(df_agg.columns.names) > 0:
            df_agg.columns = df_agg.columns.set_names(
                [
                    None,
                ]
                * len(df_agg.columns.names)
            )

        if row_order is not None:
            df_agg = df_agg.loc[row_order]
        style = df_agg.style.format(
            formatter,
            precision=self.n_decimals,
        )

        # Apply the highlight function to the specified columns (only if not merged)
        if not self.merge_aggs:
            for i in range(1, k + 1):
                style.apply(
                    partial(
                        self._find_k_th_fn, fn=highlight_fn, k=i, props=props[i - 1]
                    ),
                    subset=([c for c in df_agg.columns if c[-1] == self.main_agg]),
                )
        else:
            # For merged aggs, apply highlighting based on pre-computed highlight_styles
            if highlight_styles:

                def apply_highlight_to_merged(x: pd.Series) -> pd.Series:
                    result = [""] * len(x)
                    row_idx = x.name
                    if row_idx in highlight_styles:
                        for col_idx, col in enumerate(x.index):
                            if col in highlight_styles[row_idx]:
                                result[col_idx] = highlight_styles[row_idx][col]
                    return result

                style.apply(apply_highlight_to_merged, axis=1)

        # Apply color mapping to all cells if color_mapping is provided
        if colors_df_before_merge is not None:
            if self.merge_aggs:
                # For merged aggs, apply colors directly from merged colors df
                style.apply(lambda x: colors_df_before_merge.loc[x.name].values, axis=1)
            else:
                # For non-merged, use the original logic
                style.apply(lambda x: colors_df_before_merge.loc[x.name].values, axis=1)

        if self.hide_agg_labels and not self.merge_aggs:
            style = style.hide(axis="columns", level=df_agg.columns.nlevels - 1)
        return style

    def get_latex(
        self,
        style: Styler,
        cols_sep: Union[str, int, None] = 0,
        n_first_cols: int | None = None,
        column_format: str | None = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Returns the LaTeX representation of the styled dataframe.

        :param style: The styled dataframe to be converted to LaTeX.
        :param cols_sep: The column separator to use.
        :param kwargs: Additional arguments to be passed to the LaTeX conversion.

        :return: The LaTeX representation of the styled dataframe.
        """
        if "convert_css" in kwargs and not kwargs["convert_css"]:
            print(
                "Warning: 'convert_css' has to be set to True for most cases. Setting to True."
            )
            del kwargs["convert_css"]
        if cols_sep is not None:
            if "column_format" in kwargs:
                del kwargs["column_format"]
            column_format = ""
            if isinstance(cols_sep, int):
                for idx in style.data.index.names:
                    print(idx)
                    column_format += "c|"
                # cols reprents the level of the multindex we want to separate
                assert cols_sep <= style.data.columns.nlevels - 1
                prev_col_name = "************"
                for i_c, c in enumerate(style.data.columns):
                    if c[cols_sep] != prev_col_name:
                        column_format += "|c"
                        prev_col_name = c[cols_sep]
                    else:
                        column_format += "c"

            elif isinstance(cols_sep, str):
                column_format = cols_sep
        latex: str = style.to_latex(
            convert_css=True,
            column_format=column_format,
            **kwargs,
        )
        return latex

    def latex(
        self,
        style: Styler,
        cols_sep: Union[str, int, None] = 0,
        **kwargs: Any,
    ) -> str:
        """
        Saves the styled dataframe to a LaTeX file.

        :param style: The styled dataframe to be saved.
        :param filename: The name of the LaTeX file.
        """
        latex = self.get_latex(style, cols_sep, **kwargs)
        return latex
