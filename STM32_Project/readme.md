# STM32 ECG Inference Project

This folder contains the STM32 project for ECG classification inference on **STM32F103ZET6**.

## Hardware target

- MCU: `STM32F103ZET6`
- Board family: ALIENTEK STM32F103 development board
- Storage: SD card
- Toolchain: Keil MDK

## What this project does

This project performs **single-sample inference** on STM32.

Workflow:

1. Initialize board peripherals
2. Mount FATFS / SD card
3. Load model weights from SD card text files
4. Load one ECG input sample from SD card
5. Run inference
6. Print 5-class logits and predicted class through serial output

## Model I/O

### Input

- Input length: `300`
- Input file: `0:/ECGData/X_test.txt`
- Format: **one integer per line**
- These values are quantized input values and will be divided by `256.0` before inference

Example:

```text
12
15
-8
23
...
