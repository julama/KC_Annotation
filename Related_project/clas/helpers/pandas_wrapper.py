"""Pandas method wrappers for flexible pandas operations in Kedro pipelines."""

import pandas as pd
from typing import Dict, Any, Union


def _handle_concat_union_cols(resolved_args: list, resolved_kwargs: dict) -> pd.DataFrame:
    """Performs pd.concat with special logic to only union new columns."""
    if not resolved_args or not isinstance(resolved_args[0], list):
        raise ValueError(
            "`concat_method: 'union_cols'` requires a list of DataFrames as the first positional argument."
        )

    dataframes = resolved_args[0]
    if not all(isinstance(df, pd.DataFrame) for df in dataframes):
        raise TypeError("All items for concatenation with 'union_cols' must be DataFrames.")

    if not dataframes:
        return pd.DataFrame()

    # This logic is specifically for column-wise concatenation.
    # It will override any user-provided 'axis' kwarg.
    resolved_kwargs['axis'] = 1

    # Build a list of dataframes for a single concat call
    dfs_to_concat = [dataframes[0]]
    existing_cols = set(dataframes[0].columns)
    for dataframe in dataframes[1:]:
        new_cols = [col for col in dataframe.columns if col not in existing_cols]
        if new_cols:
            dfs_to_concat.append(dataframe[new_cols])
            existing_cols.update(new_cols)

    return pd.concat(dfs_to_concat, **resolved_kwargs)


def _handle_merge_asof_sorted(resolved_args: list, resolved_kwargs: dict) -> pd.DataFrame:
    """Performs pd.merge_asof with pre-sorting of dataframes."""
    if len(resolved_args) != 2 or not all(isinstance(df, pd.DataFrame) for df in resolved_args):
        raise ValueError("`sort_before_merge` for `merge_asof` requires exactly 2 DataFrames as positional arguments.")

    left_df, right_df = resolved_args

    # --- Inlined logic from _sort_dataframes_for_merge ---
    # Create copies to avoid modifying original DataFrames
    left_sorted = left_df.copy()
    right_sorted = right_df.copy()

    # Sort left DataFrame
    if resolved_kwargs.get('left_index', False):
        left_sorted.sort_index(inplace=True)
    elif resolved_kwargs.get('left_on'):
        left_sorted.sort_values(by=resolved_kwargs['left_on'], inplace=True)
    elif resolved_kwargs.get('on'):
        left_sorted.sort_values(by=resolved_kwargs['on'], inplace=True)
    else:
        raise ValueError("No sort column specified for left DataFrame. Use 'left_on', 'on', or 'left_index'")

    # Sort right DataFrame
    if resolved_kwargs.get('right_index', False):
        right_sorted.sort_index(inplace=True)
    elif resolved_kwargs.get('right_on'):
        right_sorted.sort_values(by=resolved_kwargs['right_on'], inplace=True)
    elif resolved_kwargs.get('on'):
        right_sorted.sort_values(by=resolved_kwargs['on'], inplace=True)
    else:
        raise ValueError("No sort column specified for right DataFrame. Use 'right_on', 'on', or 'right_index'")

    return pd.merge_asof(left_sorted, right_sorted, **resolved_kwargs)

def pandas_sequence_wrapper(*args) -> Union[pd.DataFrame, Any]:
    """
    Wrapper for executing a sequence of pandas functions and general object methods.

    This function can chain operations, passing the result of one step to the next.
    It supports both pandas top-level functions (e.g., `pd.merge`) and methods
    on any object, such as a DataFrame, Series, or GroupBy object.

    To distinguish pandas functions, prefix the method name with "pd.".
    - `method`: "pd.merge" -> `pd.merge()`
    - `method": "query"    -> `some_object.query()`

    Args:
        *args: Variable arguments where:
            - args[0] to args[n-2]: Input DataFrames or other initial objects.
            - args[n-1]: A list of operation dictionaries.

    Returns:
        The final result after applying all operations.

    Referencing Objects:
    - Initial inputs can be referenced by `df0`, `df1`, etc. (even if not DataFrames).
    - The result of the previous step can be referenced by `__result__`.
    - For object methods, the `on` key specifies which object to call the method on
      (defaults to `__result__`).
    - Simple attribute access is supported using dot notation (e.g., `"df1.index"`).

    Storing Intermediate Results:
    - Use the `output_key` parameter to store an operation's result.
    - This stored result can be referenced by its key in subsequent steps.

    Special Functionality:
    - `pd.concat`:
        To concatenate DataFrames ensuring no duplicate column names (a "column union"),
        use the special `concat_method` kwarg. The wrapper takes all columns from the
        first DataFrame and only new, non-overlapping columns from subsequent ones.
        Example:
            {
                "method": "pd.concat",
                "args": [["df0", "df1", "df2"]],
                "kwargs": {"concat_method": "union_cols"}
            }

    - `pd.merge_asof`:
        To ensure dataframes are sorted on the merge key before the operation (a
        requirement for `merge_asof`), use the `sort_before_merge` kwarg.
        Example:
            {
                "method": "pd.merge_asof",
                "args": ["df_left", "df_right"],
                "kwargs": {
                    "on": "timestamp",
                    "by": "patient_id",
                    "sort_before_merge": True
                }
            }

    - `loc` and `iloc`:
        For indexing operations, use the `indexer` parameter to specify what to index with.
        The `on` parameter specifies which object to index (defaults to `__result__`).
        Simple attribute access like "df1.index" is supported.
        Examples:
            # One-dimensional indexing: df0.loc[df1.index]
            {
                "method": "loc",
                "on": "df0",
                "indexer": "df1.index"
            }
            
            # One-dimensional indexing with boolean mask
            {
                "method": "loc",
                "on": "df0",
                "indexer": "boolean_mask"
            }
            
            # Two-dimensional indexing: df0.loc[row_mask, ['col1', 'col2']]
            {
                "method": "loc", 
                "on": "df0",
                "indexer": ["row_mask", ["col1", "col2"]]
            }
            
            # Using iloc for positional indexing (first 5 rows)
            {
                "method": "iloc",
                "on": "df0", 
                "indexer": slice(0, 5)
            }

    Example for a standard pd.concat:
        [
            {
                "method": "pd.concat",
                "args": [["df0", "df1"]],
                "kwargs": {"axis": 1}
            }
        ]
    This is equivalent to: pd.concat([df0, df1], axis=1)
    """
    params = args[-1]
    initial_objects = args[:-1]

    if not initial_objects:
        raise ValueError("pandas_sequence_wrapper requires at least one input object.")

    # Context holds all original objects and intermediate results.
    context = {f"df{i}": obj for i, obj in enumerate(initial_objects)}
    initial_obj_keys = list(context.keys())
    context["__result__"] = initial_objects[0]

    def resolve_arg(arg):
        """Recursively resolves string references to their objects from context."""
        if isinstance(arg, str):
            # Handle simple attribute access like "df1.index"
            if '.' in arg:
                parts = arg.split('.')
                if len(parts) == 2:
                    obj_key, attr_name = parts
                    if obj_key in context:
                        try:
                            return getattr(context[obj_key], attr_name)
                        except AttributeError:
                            # If attribute doesn't exist, return the original string
                            return arg
            # Simple object reference
            return context.get(arg, arg)
        if isinstance(arg, list):
            return [resolve_arg(x) for x in arg]
        if isinstance(arg, dict):
            # For kwargs, we only want to resolve values, not keys.
            return {k: resolve_arg(v) for k, v in arg.items()}
        return arg

    for operation in params:
        method_name = operation["method"]
        op_args = operation.get("args", [])
        op_kwargs = operation.get("kwargs", {})
        output_key = operation.get("output_key")

        print('pandas_sequence_wrapper: operation', operation)

        #for key, obj in context.items():
        #    print(key, obj.columns)

        # Resolve string references in args and kwargs to actual objects
        resolved_args = resolve_arg(op_args)
        resolved_kwargs = resolve_arg(op_kwargs)

        if method_name.startswith("pd."):
            # It's a pandas function (e.g., "pd.merge")
            func_name = method_name[3:]

            # Special handling for concat with column union
            if func_name == "concat" and resolved_kwargs.pop("concat_method", None) == "union_cols":
                new_result = _handle_concat_union_cols(resolved_args, resolved_kwargs)
            
            # Special handling for merge_asof sorting
            elif func_name == "merge_asof" and resolved_kwargs.pop("sort_before_merge", False):
                new_result = _handle_merge_asof_sorted(resolved_args, resolved_kwargs)

            else:
                # Generic call for all other pd functions
                pandas_func = getattr(pd, func_name)
                new_result = pandas_func(*resolved_args, **resolved_kwargs)
        
        elif method_name in ["loc", "iloc"]:
            # Special handling for indexing operations like .loc[] and .iloc[]
            on_key = operation.get("on", "__result__")
            target_obj = resolve_arg(on_key)
            
            indexer = operation.get("indexer")
            if indexer is None:
                raise ValueError(f"'{method_name}' operation requires an 'indexer' parameter.")
            
            resolved_indexer = resolve_arg(indexer)
            
            # Get the indexer object (.loc or .iloc)
            indexer_obj = getattr(target_obj, method_name)
            
            # Handle different indexing patterns
            if isinstance(resolved_indexer, (list, tuple)) and len(resolved_indexer) == 2:
                # Two-dimensional indexing: df.loc[row_indexer, col_indexer]
                new_result = indexer_obj[resolved_indexer[0], resolved_indexer[1]]
            else:
                # One-dimensional indexing: df.loc[indexer]
                new_result = indexer_obj[resolved_indexer]
        
        else:
            # It's a regular method on an object (e.g., "query" on a DF, "agg" on a GroupBy)
            on_key = operation.get("on", "__result__")
            target_obj = resolve_arg(on_key)

            try:
                method = getattr(target_obj, method_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Object of type '{type(target_obj).__name__}' (from key '{on_key}') "
                    f"has no attribute '{method_name}'"
                ) from e

            new_result = method(*resolved_args, **resolved_kwargs)

        # Update the __result__ for the next iteration
        context["__result__"] = new_result

        # If an output_key is provided, store the result in the context.
        if output_key:
            if not isinstance(output_key, str):
                raise TypeError(f"'output_key' must be a string, not {type(output_key)}.")
            if output_key in initial_obj_keys:
                raise ValueError(f"'{output_key}' is a reserved key for an initial object.")
            if output_key == "__result__":
                raise ValueError("'__result__' is a reserved key and cannot be used as an output_key.")
            context[output_key] = new_result

    return context["__result__"]