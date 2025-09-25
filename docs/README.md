## Usage Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Oceana_GCast
   cd Oceana_GCast
   ```

2. **Install Dependencies**:
   Ensure you have Julia installed, then run:
   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```

3. **Run the Simulation**:
   Execute the main simulation script:
   ```bash
   julia scripts/run_simulation.jl
   ```

4. **Preprocess Data**:
   To preprocess input data, run:
   ```bash
   julia scripts/preprocess_data.jl
   ```

5. **Run Tests**:
   To ensure everything is working correctly, run the tests:
   ```bash
   julia --project -e 'using Pkg; Pkg.test()'
   ```

## Contribution

Contributions are welcome! Please feel free to submit issues or pull requests to enhance the functionality of Oceana_GCast.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
