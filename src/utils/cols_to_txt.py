def read_columns_from_file(filename='columns.txt'):
    columns = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Strip whitespace and add to list if not empty
                column = line.strip()
                if column:
                    columns.append(column)
        print(f"Successfully read {len(columns)} columns from {filename}")
        return columns
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def save_columns_to_file(columns, filename='columns.txt'):
    with open(filename, 'w') as f:
        for column in columns:
            f.write(f"{column}\n")
    print(f"Columns saved to {filename}")



if __name__ == "__main__":
    selected_cols = ['lon_rounded', 'lat_rounded', 't2m_lag_3', 't2m_lag_7', 't2m_lag_14', 't2m_lag_30',
                    't2m_lag_45', 't2m_lag_60', 't2m_lag_90', 't2m_lag_120', 't2m_std_7', 't2m_sum_7',
                    't2m_sum_14', 't2m_std_30', 't2m_mean_45', 't2m_sum_45', 't2m_mean_60', 't2m_mean_90',
                    't2m_std_90', 't2m_ewm_7', 't2m_ewm_60', 't2m_diff_30', 't2m_diff_45', 't2m_diff_60',
                    't2m_diff_120', 't2m_pct_change_7', 't2m_pct_change_14', 't2m_pct_change_30',
                    't2m_pct_change_90', 't2m_max_7', 't2m_min_14', 't2m_max_14', 't2m_median_45',
                    't2m_max_60', 't2m_max_90', 't2m_median_90', 't2m_min_120', 't2m_median_120',
                    't2m_autocorr_14', 't2m_autocorr_45', 't2m_autocorr_120', 't2m_fft_peak_1',
                    't2m_fft_peak_3', 't2m_trend_slope_7', 't2m_trend_intercept_14', 't2m_trend_intercept_30',
                    't2m_trend_intercept_45', 't2m_trend_slope_60', 't2m_trend_slope_90', 't2m_trend_intercept_90',
                    't2m_trend_slope_120', 'tp_mean_7', 'tp_std_7', 'tp_mean_14', 'tp_sum_14', 'tp_sum_45',
                    'tp_mean_60', 'tp_sum_90', 'tp_std_120', 'tp_sum_120', 'tp_min_7', 'tp_max_7', 'tp_median_7',
                    'tp_min_14', 'tp_max_14', 'tp_min_30', 'tp_max_45', 'tp_median_45', 'tp_min_60', 'tp_median_60',
                    'tp_median_90', 'tp_max_120', 'tp_fft_peak_2', 'tp_cumsum', 'tp_cumprod', 'tp_trend_slope_7',
                    'tp_trend_slope_14', 'tp_trend_intercept_14', 'tp_trend_slope_30', 'tp_trend_intercept_45',
                    'tp_trend_slope_60', 'tp_trend_slope_120', 'd2m_mean_14', 'd2m_sum_14', 'd2m_mean_30', 'd2m_mean_60',
                    'd2m_std_60', 'd2m_sum_60', 'd2m_mean_120', 'd2m_sum_120', 'd2m_ewm_7', 'd2m_ewm_14', 'd2m_ewm_30',
                    'd2m_ewm_90', 'd2m_ewm_120', 'd2m_diff_3', 'd2m_diff_7', 'd2m_diff_14', 'd2m_diff_90', 'd2m_diff_120',
                    'd2m_pct_change_3', 'd2m_pct_change_7', 'd2m_pct_change_60', 'd2m_pct_change_90', 'd2m_pct_change_120',
                    'd2m_min_7', 'd2m_max_7', 'd2m_min_14', 'd2m_median_14', 'd2m_min_30', 'd2m_median_30', 'd2m_max_45',
                    'd2m_median_45', 'd2m_min_60', 'd2m_min_90', 'd2m_median_90', 'd2m_max_120', 'd2m_autocorr_3',
                    'd2m_autocorr_14', 'd2m_autocorr_60', 'd2m_autocorr_90', 'd2m_fft_peak_1', 'd2m_fft_peak_2',
                    'd2m_trend_slope_7', 'd2m_trend_slope_30', 'd2m_trend_intercept_30', 'd2m_trend_intercept_60',
                    'd2m_trend_intercept_90', 'elevation_point_stddev', 'elevation_point_min', 'elevation_min_0.1deg',
                    'elevation_max_0.1deg', 'elevation_mean_0.1deg', 'elevation_std_0.1deg', 'elevation_gradient_max_0.1deg',
                    'elevation_gradient_std_0.1deg', 'elevation_min_0.15deg', 'elevation_max_0.15deg', 'elevation_std_0.15deg',
                    'elevation_gradient_min_0.15deg', 'elevation_gradient_std_0.15deg', 'elevation_max_0.25deg', 'elevation_std_0.25deg',
                    'elevation_gradient_min_0.25deg', 'elevation_gradient_std_0.25deg',
                    'road_density_10', 'road_density_50', 'road_density_100', 'datetime', 'day']
    save_columns_to_file(selected_cols)