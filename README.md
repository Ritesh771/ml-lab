
```markdown
# ML Lab Programs

This repository contains machine learning lab programs. Below are the instructions to set up and run the programs in different environments.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.11 or later
- pip (Python package installer)

## Setup and Running Instructions

### Running on VS Code

1. **Create a Virtual Environment**:
   Open your terminal and navigate to the project directory. Run the following command to create a virtual environment:

   ```bash
   python3.11 -m venv env
   ```

2. **Activate the Virtual Environment**:
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```

3. **Install Required Packages**:
   With the virtual environment activated, install the necessary packages:

   ```bash
   pip install scikit-learn numpy pandas matplotlib seaborn
   ```

4. **Run the Program**:
   Execute the Python script `pr1.py`:

   ```bash
   python pr1.py
   ```

### Running on Jupyter Notebook

1. **Install Required Packages**:
   In a Jupyter Notebook cell, run the following command to install the necessary packages:

   ```python
   !pip install scikit-learn numpy pandas matplotlib seaborn
   ```

2. **Run the Notebook**:
   Open your Jupyter Notebook and run the cells sequentially to execute the machine learning lab programs.

## Additional Information

- Ensure you have the necessary datasets in the correct directory or update the file paths in the scripts as needed.
