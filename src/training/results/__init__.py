# File management suite for the training output
# Mostly just CSVs and Pyplot images

from datetime import datetime
import os

import pandas as pd
import matplotlib.pyplot as plt


# All files will be identified by a timestamp, which is determined when this package is imported
TIMESTAMP = datetime.now().isoformat().replace('T', '_').replace(':', '-').split('.')[0]

filename_auto_stamp = f'{os.path.dirname(__file__)}/{TIMESTAMP}_{"{}"}.{"{}"}'
filename_complete = f'{os.path.dirname(__file__)}/{"{}"}.{"{}"}'


# Save the results to CSV
# Results should be stored in a dictionary with meaningful key names
def save(results, name):
    df = pd.DataFrame(results)
    df.to_csv(filename_auto_stamp.format(name, 'csv'))


# Load results of previous training from CSV
# Could be useful if you want to continue a traning session from before
def load(name):
    return pd.read_csv(filename_complete.format(name, 'csv')).to_dict(orient='list')

# Create a pyplot of the results
# cols_to_plot should be a list of strings, or the string 'all'
def create_pyplot(results, cols_to_plot, name):
    if cols_to_plot == 'all':
        cols_to_plot = results.keys()

    for col in cols_to_plot:
        if len(results[col]) > 0:
            plt.plot(results[col], label=col)

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(filename_auto_stamp.format(name, 'png'), bbox_inches='tight', dpi=250)
    plt.close()


if __name__ == "__main__":
    print(TIMESTAMP)