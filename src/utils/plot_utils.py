import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

country_mapping = {
        'Russian_Federation': 'Russia',
        'United_Kingdom': 'United Kingdom',
        'Czech_Republic': 'Czechia',
        'Bosnia_and_Herzegovina': 'Bosnia and Herzegovina',
        'Serbia': 'Republic of Serbia',
    }


def plot_target_data(data: pd.DataFrame, countries: list, save_path: str = None, points_size: int = 1, 
                    lon_range = None):
    """
    Plot the target data points overlaid on country borders
    
    Args:
        data (pd.DataFrame): The prepared target data
        countries (list): List of countries to plot
        save_path (str, optional): Path to save the plot. If None, displays the plot
        points_size (int, optional): Size of the scatter plot points
        lon_range (tuple, optional): Tuple of (min_longitude, max_longitude) to display. Default is (0, 360)
    """
    world = gpd.read_file('data/countries')
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    
    for country in countries:
        search_country = country_mapping.get(country, country)
        country_data = world[world['SOVEREIGNT'] == search_country]
        if len(country_data) > 0:
            country_data.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)

    if lon_range is None:
        filtered_data = data
        lon_range = (data['lon_rounded'].min(), data['lon_rounded'].max())
    else:
        
        lon_mask = (data['lon_rounded'] >= lon_range[0]) & (data['lon_rounded'] <= lon_range[1])
        filtered_data = data[lon_mask]

    print("Longitude range: ", lon_range)
    
    lon_converted = filtered_data['lon_rounded']
    
    positive_samples = filtered_data[filtered_data['count'] > 0]
    ax.scatter(lon_converted[filtered_data['count'] > 0], positive_samples['lat_rounded'], 
              c='red', s=points_size, alpha=0.4, label='Fire locations')
    
    negative_samples = filtered_data[filtered_data['count'] == 0]
    ax.scatter(lon_converted[filtered_data['count'] == 0], negative_samples['lat_rounded'], 
              c='blue', s=points_size, alpha=0.05, label='Negative samples')
    
    ax.set_title('Target Data Distribution')
    ax.set_xlim(lon_range[0], lon_range[1])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize




def plot_precision_recall_curve(y_true, y_pred_proba, n_classes=None):
    """
    Plot Precision-Recall curve for both binary and multiclass classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    n_classes : int, optional
        Number of classes. If None, inferred from data.
    """
    # If number of classes not specified, infer from data
    if n_classes is None:
        n_classes = len(np.unique(y_true))
    
    # Binarize the output for multiclass or ensure correct format
    if n_classes > 2:
        # Multiclass case
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute Precision-Recall curve and average precision for each class
        plt.figure(figsize=(10, 8))
        
        # Store precision, recall, average precision for each class
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            avg_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            plt.plot(recall[i], precision[i], 
                     label=f'Precision-Recall curve (class {i}, AP = {avg_precision[i]:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Multiclass')
        plt.legend(loc="lower left")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
    
    else:
        # Binary classification case
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', 
                 label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        
        # Plot the baseline (random classifier)
        plt.plot([0, 1], [np.mean(y_true), np.mean(y_true)], 'r--', 
                 label='Baseline (Random Classifier)')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Binary Classification')
        plt.legend(loc="lower left")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
    
    plt.tight_layout()
    plt.show()


def visualize_column_on_map(df, column, start_time, end_time, time_column='datetime'):
    """
    Visualizes a specified column from the dataframe on a map using
    the 'lat_rounded' and 'lon_rounded' columns, after filtering the data
    based on a start and end time.

    Parameters:
      df         : pandas.DataFrame
                   The DataFrame containing the features including the 
                   geographical coordinates as 'lat_rounded' and 'lon_rounded'
                   and a time column.
      column     : str
                   The name of the column to visualize.
      start_time : str or datetime-like
                   The lower time bound (e.g., '2020-01-01').
      end_time   : str or datetime-like
                   The upper time bound (e.g., '2020-12-31').
      time_column: str, default 'datetime'
                   The name of the time column present in df.

    The function uses Plotly Express to create a scatter mapbox plot.
    """
    import pandas as pd
    import plotly.express as px

    # Ensure the time column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])

    # Filter the dataframe based on the provided time bounds
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    df_filtered = df[(df[time_column] >= start_time) & (df[time_column] <= end_time)]

    # Create a scatter mapbox visualization
    fig = px.scatter_mapbox(
        df_filtered,
        lat="lat_rounded",
        lon="lon_rounded",
        color=column,
        hover_data=[time_column, column],
        zoom=3,
        height=600,
        title=f"{column} visualization between {start_time.date()} and {end_time.date()}"
    )

    # Use an open-source base map style
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0, "t":30, "l":0, "b":0})
    fig.show()
