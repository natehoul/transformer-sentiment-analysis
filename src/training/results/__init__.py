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
    df.to_csv(filename_auto_stamp.format(name, 'csv'), index=False)


# Load results of previous training from CSV
# Could be useful if you want to continue a traning session from before
def load(name):
    return pd.read_csv(filename_complete.format(name, 'csv')).to_dict(orient='list')

# Create a pyplot of the results
# cols_to_plot should be a list of strings, or the string 'all'
def create_pyplot(results, cols_to_plot, name, normalize_loss=False):
    if cols_to_plot == 'all':
        cols_to_plot = results.keys()

    training = [col for col in cols_to_plot if col[:8] == 'Training' and len(results[col]) > 0]
    validation = [col for col in cols_to_plot if col[:10] == 'Validation' and len(results[col]) > 0]
    loss_cols = ['Training Loss', 'Validation Loss']

    if normalize_loss:
        # Normalize the losses to the range [0, 1]
        max_loss = max([max(results[loss_col]) for loss_col in loss_cols if loss_col in cols_to_plot])
        for loss_col in loss_cols:
            if loss_col in cols_to_plot:
                results[loss_col] = [loss / max_loss for loss in results[loss_col]]

    # Each type of data (loss, accuracy, precision, recall, f1) has a unique color
    color_order = ['red', 'green', 'blue', 'purple', 'black']

    # Each data group (training, validation) has a unique line style
    line_style = ['dashed', 'solid']

    for result_type, style in zip((training, validation), line_style):
        for col, color in zip(result_type, color_order):
            label = col
            if col in loss_cols and normalize_loss:
                label += ' (Normalized)'

            plt.plot(results[col], label=label, linestyle=style, color=color, linewidth=0.75)

    # All data is in the range [0, 1]
    plt.ylim(-0.05, 1.05)

    # The legend will be outside the plot area
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

    
    plt.savefig(filename_auto_stamp.format(name, 'png'), bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    name = "2022-04-19_16-04-47_music_rating"
    results = load(name)
    #save(results, "music_rating")
    create_pyplot(results, 'all', name)