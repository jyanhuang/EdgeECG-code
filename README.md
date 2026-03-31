# EdgeECG

This repository provides the official implementation of **EdgeECG** for arrhythmia classification on resource-constrained edge devices.

The repository includes:

- Python scripts for data preparation and deployment file generation
- C implementation for PC-side inference validation
- STM32 deployment project for embedded inference

---

## Repository Structure

- `Python_Implementation/`  
  Python scripts for preprocessing, dataset generation, and export of model-related text files.

- `C_Implementation/`  
  C implementation for PC-side validation of the inference pipeline, including:
  - parameter loading from txt files
  - test input loading
  - classification result verification

- `STM32_Project/`  
  Embedded deployment project targeting **STM32F103ZET6**.
---
