import pandas as pd
from hampel import hampel

# Sample data as a pandas.Series
data = pd.Series([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])

# Apply the Hampel filter
result = hampel(data, window_size=3, n_sigma=3.0)

print(result.filtered_data)
print(result.outlier_indices)
print(result.medians)
print(result.median_absolute_deviations)
print(result.thresholds)
