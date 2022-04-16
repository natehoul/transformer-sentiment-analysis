import os

# Dictionary containing all of the filenames (w/o extension) and dataset sizes of the data stored in this directory
# Keys should all be lower case
raw_data = \
    {
        'tools_and_home_improvement': ('Tools_and_Home_Improvement_5', 2_070_831),
        'musical_instruments': ('Musical_Instruments_5', 231_392)
    }

# A dictionary containing all of the different ways you can refer to a given dataset
# Keys should match a key in raw_data
aliases = \
    {
        'tools_and_home_improvement': {'thi', 'tahi', 'tools'},
        'musical_instruments': {'music', 'mi', 'instruments'}
    }


# The possible extensions to try, in typical priority order
extensions = ['.pkl', '.json.gz']


# Converts an alias to its true key name
# Returns None if key not found
def resolve_alias(alias: str) -> str or None:
    alias = alias.lower()
    if alias in raw_data:
        return alias

    else:
        for key, alias_set in aliases.items():
            if alias in alias_set:
                return key
    
    return None


# Input:
#   alias (str): One alias for a raw dataset
#   prefer_pickle (bool): True --> The funciton will attempt to return pickled data, and return .json.gz only if it can't
#                         False --> The function will attempt to return .json.gz data, and pickled if it can't
# Output:
#   is_pickle (bool): True --> path is a path to a pickle file; False --> path is a path to a .json.gz file 
#   path (str): The path to the file with the data
#   size (int): The number of data instances within the file
# Raises:
#   FileNotFoundError, if it couldn't find either file (neither pickle nor json.gz)
# Purpose: 
#   While you could just use raw_data[s], that makes aliasing harder; I like aliases
#   This also enforces all lower case, which makes things easier
#   Also dynamically selects the pickle data over the json.gz data, unless otherwise specified
def get_raw_data(alias: str, prefer_pickle: bool=True):
    key = resolve_alias(alias)

    filename, size = raw_data[key]
    filename = os.path.dirname(__file__) + '/' + filename
    extension_order = extensions[::1] if prefer_pickle else extensions[::-1]
    extension_order.append(None)

    for extension in extension_order:
        if extension is None:
            raise FileNotFoundError(f'Neither .json.gz nor .pkl exists for {key}')

        path = filename + extension
        if os.path.isfile(path):
            break

    is_pickle = os.path.splitext(path)[1] == '.pkl'

    return is_pickle, path, size