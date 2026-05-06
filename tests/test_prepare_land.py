import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_generation.prepare_land import get_elevation_stats, landsea_distance

def test_generate_elevation_maps_for_region():
    """
    Generates and saves visual maps of elevation features for a specific
    geographic region (Finland and Sweden) to allow for visual inspection.
    """
    print("Starting test: Generate elevation feature maps for Finland and Sweden.")

    # 1. Define region and create grid points
    # Bounding box for Finland and Sweden
    lat_min, lat_max = 55, 71
    lon_min, lon_max = 10, 33
    resolution = 0.1  # Degrees

    lats = np.arange(lat_min, lat_max, resolution)
    lons = np.arange(lon_min, lon_max, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    target_df = pd.DataFrame({
        'lat_rounded': lat_grid.flatten(),
        'lon_rounded': lon_grid.flatten()
    })

    print(f"Generated a grid of {len(target_df)} points for the test region.")

    # 2. Define parameters for get_elevation_stats
    nc_path = "data/land_features/topography.nc"
    if not os.path.exists(nc_path):
        print(f"Error: Topography data file not found at {nc_path}. Skipping test.")
        return

    window_sizes = [0.05, 0.25, 0.5]
    # Variables to process from the topography file
    topo_vars = ["z", "sdor", "sdfor", "anor", "isor", "slor"]

    output_dir = "tests/output"
    print(f"Saving feature maps to directory: {output_dir}")

    # 3. Call the function for each variable and generate plots
    for var in topo_vars:
        print(f"\nProcessing variable: '{var}'")
        try:
            stats_df, feature_names = get_elevation_stats(
                nc_path=nc_path,
                target_df=target_df,
                window_sizes=window_sizes,
                elevation_var=var
            )
            print(f"  Stats calculated for '{var}'.")

            # Remove coordinate columns from feature names for plotting
            plot_features = [
                col for col in feature_names if col not in ['latitude', 'longitude']
            ]
            
            # 4. Generate and save plots for each feature
            for feature in plot_features:
                if feature in stats_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 12))
                    
                    # Reshape the flat feature data back into a 2D grid for plotting
                    data_grid = stats_df[feature].values.reshape(len(lats), len(lons))
                    
                    # Create the plot
                    im = ax.imshow(
                        data_grid,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin='lower',
                        cmap='viridis'
                    )
                    
                    plt.colorbar(im, ax=ax, label=feature)
                    ax.set_title(f'Feature Map: {feature}')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    
                    # Save the figure
                    plot_path = os.path.join(output_dir, f'map_{feature}.png')
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  - Saved map for '{feature}' to {plot_path}")
                else:
                    print(f"  Warning: Feature '{feature}' not found in the output DataFrame.")
        except KeyError as e:
            print(f"  Could not process variable '{var}'. Skipping. Reason: {e}")

    print("\nTest completed. Check the 'tests/output' directory for map images.")


def test_landsea_distance_binarizes_non_binary_mask(tmp_path):
    mask_path = tmp_path / "mask.nc"
    dist_path = tmp_path / "distances.nc"
    mask = xr.Dataset(
        {
            "landseamask": (
                ("lat", "lon"),
                np.array(
                    [
                        [100.0, 100.0, 100.0],
                        [100.0, 1.0, 100.0],
                        [100.0, 100.0, 100.0],
                    ],
                    dtype=np.float32,
                ),
            )
        },
        coords={"lat": np.array([0.0, 1.0, 2.0]), "lon": np.array([0.0, 1.0, 2.0])},
    )
    mask.to_netcdf(mask_path)
    points = pd.DataFrame(
        {
            "lat_rounded": [1.0, 0.0],
            "lon_rounded": [1.0, 0.0],
        }
    )

    distances, feature_names = landsea_distance(
        points,
        mask_path=str(mask_path),
        dist_path=str(dist_path),
    )

    assert feature_names == ["distance_to_coast_km", "distance_to_coast_dilated_km"]
    assert distances.loc[0, "distance_to_coast_km"] > 0
    assert distances.loc[1, "distance_to_coast_km"] < 0

if __name__ == "__main__":
    test_generate_elevation_maps_for_region()
