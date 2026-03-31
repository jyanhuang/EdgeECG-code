# C Implementation for PC Validation

This folder contains the C implementation used for **PC-side validation** of the ECG model before STM32 deployment.

## 1. Files

This folder includes:

- `main.c`  
  PC-side inference source code
- `main.exe`  
  Compiled executable for Windows
- `ECGData/`  
  Folder containing model parameter text files
- `X_test_np.zip`  
  Compressed ECG test input file
- `Y_test_np.txt`  
  Label file for the test set
- `Results.png`  
  Example validation result
- `README.md`

### Important note

Please unzip:

```text
X_test_np.zip
```
to obtain:
```text
X_test_np.txt
```
before running the program.

## 2. Environment Requirements

Before running this project, please make sure the following environment is prepared:

- **Operating System:** Windows
- **Compiler:** GCC / MinGW-w64 or Visual Studio C/C++ Build Tools
- **Build Tool:** CMake
- **Editor (optional):** VS Code

## 3. Run the Project

### Method 1: Run the executable directly

```bash
.\main.exe
```
### Method 2: Rebuild and run
```bash
gcc main.c -o main.exe
.\main.exe
```
