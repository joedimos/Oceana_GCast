## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Julia (version 1.6 or later)
- Python (version 3.7 or later)
- pip (Python package manager)

## Installation Steps

1. **Clone the Repository**

   Open a terminal and clone the Oceananigans project repository:

   ```bash
   git clone https://github.com/yourusername/oceananigans-project.git
   cd oceananigans-project
   ```

2. **Set Up Julia Environment**

   Open Julia and activate the project environment:

   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

   This will install all necessary Julia dependencies listed in the `Project.toml` file.

3. **Install Python Dependencies**

   Navigate to the `graphcast` directory and install the required Python packages:

   ```bash
   cd graphcast
   pip install -r requirements.txt
   ```

   If you do not have a `requirements.txt` file, you can manually install the necessary packages:

   ```bash
   pip install xarray numpy jax haiku chex
   ```

4. **Download ERA5 Data**

   Ensure you have the `era5_sample.nc` file in the `data` directory. If you need to download it, refer to the ERA5 documentation for instructions on obtaining the data.

5. **Run the Simulation**

   You can run the main simulation script using Julia:

   ```bash
   julia scripts/run_simulation.jl
   ```

6. **Testing the Setup**

   To ensure everything is set up correctly, run the tests provided in the `tests` directory:

   ```bash
   julia test/test_graphcast.jl
   julia test/test_simulation.jl
   ```

## Additional Notes

- If you encounter any issues, please check the README.md file for troubleshooting tips.
- For further customization, refer to the configuration options in the `src` directory files.

## Contact

For questions or support, please open an issue in the GitHub repository or contact the project maintainers.
