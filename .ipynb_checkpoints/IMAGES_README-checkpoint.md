# 📊 Saved Visualizations - House Price Estimator

## Overview

All visualizations from the House Price Estimator Jupyter notebook are automatically saved to the `images/` directory when you run the notebook.

## Saved Images (12 Total)

| # | Filename | Description |
|---|----------|-------------|
| 1 | `01_price_distribution.png` | Distribution of house prices (histogram with KDE) |
| 2 | `02_sqft_vs_price.png` | Relationship between square footage and price (scatter plot) |
| 3 | `03_bedrooms_vs_price.png` | Average house prices by number of bedrooms (bar chart) |
| 4 | `04_correlation_heatmap.png` | Correlation matrix of all features (heatmap) |
| 5 | `05_location_vs_price.png` | House prices by geographic location (scatter plot with color) |
| 6 | `06_grade_vs_price.png` | Average price by house grade (bar chart) |
| 7 | `07_year_built_vs_price.png` | Price trends over construction year (line chart) |
| 8 | `08_waterfront_vs_price.png` | Price comparison for waterfront vs non-waterfront (box plot) |
| 9 | `09_sqft_living_distribution.png` | Distribution of living area square footage (histogram with KDE) |
| 10 | `10_bathrooms_vs_price.png` | Relationship between bathrooms and price (scatter plot) |
| 11 | `11_condition_distribution.png` | Distribution of house conditions (count plot) |
| 12 | `12_residual_plot.png` | Model residuals vs predicted values (scatter plot) |

## Image Specifications

- **Format**: PNG
- **Resolution**: 300 DPI (publication quality)
- **Color Palette**: Custom themed colors for better readability
- **Background**: Cream color (#fffdd0) for reduced eye strain

## Usage

### How Images Are Saved

When you run the notebook, each visualization cell automatically:
1. Creates the plot
2. Saves it to `images/` directory with a descriptive filename
3. Displays the plot in the notebook
4. Prints a confirmation message (e.g., "💾 Saved: images/01_price_distribution.png")

### Viewing Saved Images

All images are saved in the `images/` directory in the same folder as your notebook:

```
House_Price_Estimator/
├── House_Price_Estimator.ipynb
├── images/
│   ├── 01_price_distribution.png
│   ├── 02_sqft_vs_price.png
│   ├── ...
│   └── 12_residual_plot.png
```

### Using Images in Reports

These high-resolution images can be used in:
- Research papers
- Presentations (PowerPoint, Google Slides)
- Blog posts
- Reports and documentation
- Portfolio projects

## Re-generating Images

To regenerate all images:
1. Open `House_Price_Estimator.ipynb` in Jupyter
2. Run all cells: `Kernel > Restart & Run All`
3. All images will be automatically recreated in the `images/` directory

The images will overwrite existing files with the same names.

## Image Quality

- All images are saved at **300 DPI**, which is:
  - Suitable for printing
  - High enough for professional presentations
  - Excellent for digital displays
  - Publication-ready quality

## Technical Details

### Dependencies

The image saving functionality requires:
- `matplotlib` - for creating and saving plots
- `seaborn` - for styled visualizations
- `pandas` - for data manipulation
- `numpy` - for numerical operations

### Code Integration

Image saving is integrated into each visualization cell using:
```python
plt.savefig('images/XX_descriptive_name.png', dpi=300, bbox_inches='tight')
```

The `bbox_inches='tight'` parameter ensures:
- No whitespace is cut off
- All labels and titles are included
- Optimal framing of the visualization

---

**Note**: The `images/` directory is automatically created when you run the notebook if it doesn't already exist.

