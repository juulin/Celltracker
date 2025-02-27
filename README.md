# Fluorescence Tracker

Fluorescence Tracker is a Python project designed to process TIF stacks of fluorescence microscopy images. The project tracks individual cells over time, calculates average intensities, standard error, and other statistics, and generates a CSV file containing this data for each cell across all frames.

## Project Structure

```
fluorescence-tracker
├── src
│   ├── main.py               # Entry point of the application
│   ├── image_processor.py     # Image processing functionalities
│   ├── cell_tracker.py        # Cell tracking functionalities
│   ├── statistics.py          # Statistical calculations
│   └── data_export.py         # Data export to CSV
├── config
│   └── settings.yaml          # Configuration settings
├── tests
│   ├── __init__.py           # Marks the tests directory as a package
│   ├── test_image_processor.py # Unit tests for ImageProcessor
│   └── test_cell_tracker.py   # Unit tests for CellTracker
├── notebooks
│   └── analysis_examples.ipynb # Jupyter notebook for analysis examples
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd fluorescence-tracker
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:

```bash
python src/main.py
```

Follow the prompts to input the path to the TIF stack of images. The application will process the images, track the cells, and generate a CSV file with the results.

## Features

- Load and preprocess TIF stacks of fluorescence microscopy images.
- Track individual cells across multiple frames.
- Calculate average intensities and standard error for each tracked cell.
- Export results to a CSV file for further analysis.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.