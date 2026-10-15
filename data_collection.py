
import yfinance as yf
from pathlib import Path

# Download historical TCS stock data
data = yf.download(
    "TCS.NS",
    start="2018-01-01",
    end="2026-10-01",
    auto_adjust=False,
    progress=False
)

# Check whether data was downloaded
if data.empty:
    print("No data downloaded. Please check your internet connection.")
else:
    # Find the project's main folder
    project_folder = Path(__file__).resolve().parent.parent

    # Create the data folder if it does not exist
    data_folder = project_folder / "data"
    data_folder.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    file_path = data_folder / "tcs_stock_data.csv"
    data.to_csv(file_path)

    print("Dataset downloaded successfully!")
    print("Number of rows and columns:", data.shape)
    print("\nFirst five records:")
    print(data.head())
    print("\nSaved at:", file_path)
    