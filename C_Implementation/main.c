#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define INPUT_LEN            300
#define CONV1_OUT_CH         8
#define PARTIAL_CH           2
#define UNTOUCHED_CH         6
#define PARTIAL_LEN          75
#define PARTIAL_REMAIN_LEN   225
#define POOL1_OUT_LEN        150
#define FC1_OUT              16
#define FC2_OUT              5
#define EPSILON              1e-5f
#define WEIGHT_SCALE         256
#define MAX_LINE_LENGTH      4096

/* ===== read from .txt files ===== */
static int conv1_weight[24];
static int conv1_bias[8];

static int conv2_partial_conv_weight[12];
static int conv2_channel_conv_weight[108];

static int conv3_weight[24];
static int conv3_bias[1];

static int bn_weight[1];
static int bn_bias[1];
static int bn_running_mean[1];
static int bn_running_var[1];

static int fc1_weight[2400];
static int fc1_bias[16];

static int fc2_weight[80];
static int fc2_bias[5];

static float buf_conv1[CONV1_OUT_CH * INPUT_LEN];
static float buf_conv2[CONV1_OUT_CH * INPUT_LEN];
static float buf_conv3[INPUT_LEN];
static float buf_pool1[POOL1_OUT_LEN];
static float buf_bn[POOL1_OUT_LEN];
static float buf_fc1[FC1_OUT];
static float buf_fc2[FC2_OUT];

static float partial_in[UNTOUCHED_CH * PARTIAL_LEN];
static float partial_out[UNTOUCHED_CH * PARTIAL_LEN];

void readCSV(const char* filename, float* array, int n)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return;
    }

    char line[MAX_LINE_LENGTH];
    int i = 0;

    while (fgets(line, MAX_LINE_LENGTH, file) && i < n) {
        char* token = strtok(line, ",");
        while (token != NULL && i < n) {
            array[i] = (float)atof(token);
            i++;
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
}

void readLabelCSV(const char* filename, float* array, int n)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return;
    }

    char line[MAX_LINE_LENGTH];
    int i = 0;

    while (fgets(line, MAX_LINE_LENGTH, file) && i < n) {
        char* token = strtok(line, ",");
        while (token != NULL && i < n) {
            array[i] = (float)atof(token);
            i++;
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
}

void readTXTInt(const char* filename, int* array, int n)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return;
    }

    char line[MAX_LINE_LENGTH];
    int i = 0;

    while (fgets(line, MAX_LINE_LENGTH, file) != NULL && i < n) {
        array[i] = atoi(line);
        i++;
    }

    if (i != n) {
        printf("Warning: file %s expected %d values, got %d\n", filename, n, i);
    }

    fclose(file);
}

void load_model_weights(void)
{
    readTXTInt("./ECGData/conv1_weight.txt", conv1_weight, 24);
    readTXTInt("./ECGData/conv1_bias.txt", conv1_bias, 8);

    readTXTInt("./ECGData/conv2_partial_conv_weight.txt", conv2_partial_conv_weight, 12);
    readTXTInt("./ECGData/conv2_channel_conv_weight.txt", conv2_channel_conv_weight, 108);

    readTXTInt("./ECGData/conv3_weight.txt", conv3_weight, 24);
    readTXTInt("./ECGData/conv3_bias.txt", conv3_bias, 1);

    readTXTInt("./ECGData/bn_weight.txt", bn_weight, 1);
    readTXTInt("./ECGData/bn_bias.txt", bn_bias, 1);
    readTXTInt("./ECGData/bn_running_mean.txt", bn_running_mean, 1);
    readTXTInt("./ECGData/bn_running_var.txt", bn_running_var, 1);

    readTXTInt("./ECGData/fc1_weight.txt", fc1_weight, 2400);
    readTXTInt("./ECGData/fc1_bias.txt", fc1_bias, 16);

    readTXTInt("./ECGData/fc2_weight.txt", fc2_weight, 80);
    readTXTInt("./ECGData/fc2_bias.txt", fc2_bias, 5);
}

void npConv1D(const float* feature,
              int length,
              const int* weight,
              const int* bias,
              int in_channels,
              int out_channels,
              int kernel_size,
              int padding,
              float* result)
{
    for (int oc = 0; oc < out_channels; oc++) {
        for (int x = 0; x < length; x++) {
            float sum = 0.0f;

            if (bias != NULL) {
                sum = (float)bias[oc] / WEIGHT_SCALE;
            }

            for (int ic = 0; ic < in_channels; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int in_x = x + k - padding;
                    if (in_x >= 0 && in_x < length) {
                        int w_index = (oc * in_channels + ic) * kernel_size + k;
                        int in_index = ic * length + in_x;
                        sum += ((float)weight[w_index] * feature[in_index]) / WEIGHT_SCALE;
                    }
                }
            }

            result[oc * length + x] = sum;
        }
    }
}

void npMaxPool1D(const float* img,
                 int length,
                 int kernel_size,
                 int stride,
                 int padding,
                 int inchannel,
                 float* result)
{
    int out_length = (length + 2 * padding - kernel_size) / stride + 1;

    for (int c = 0; c < inchannel; c++) {
        for (int j = 0; j < out_length; j++) {
            int start = j * stride - padding;
            float max_val = -FLT_MAX;

            for (int k = 0; k < kernel_size; k++) {
                int idx = start + k;
                float val = -FLT_MAX;
                if (idx >= 0 && idx < length) {
                    val = img[c * length + idx];
                }
                if (val > max_val) {
                    max_val = val;
                }
            }

            result[c * out_length + j] = max_val;
        }
    }
}

void npBatchNorm1D(const float* input,
                   int length,
                   int channels,
                   const int* gamma,
                   const int* beta,
                   const int* running_mean,
                   const int* running_var,
                   float* output)
{
    for (int c = 0; c < channels; c++) {
        float g = (float)gamma[c] / WEIGHT_SCALE;
        float b = (float)beta[c] / WEIGHT_SCALE;
        float mean = (float)running_mean[c] / WEIGHT_SCALE;
        float var = (float)running_var[c] / WEIGHT_SCALE;

        float scale = g / sqrtf(var + EPSILON);
        float shift = b - mean * scale;

        for (int i = 0; i < length; i++) {
            output[c * length + i] = input[c * length + i] * scale + shift;
        }
    }
}

void Linear(const float* input_data,
            int length,
            int inchannel,
            int outnumber,
            const int* weights,
            const int* bias,
            float* output)
{
    int in_features = length * inchannel;

    for (int o = 0; o < outnumber; o++) {
        float sum = 0.0f;

        if (bias != NULL) {
            sum = (float)bias[o] / WEIGHT_SCALE;
        }

        for (int i = 0; i < in_features; i++) {
            sum += ((float)weights[o * in_features + i] * input_data[i]) / WEIGHT_SCALE;
        }

        output[o] = sum;
    }
}

void ReLU(float* data, int n)
{
    for (int i = 0; i < n; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

void Partial_conv1D(const float* input, float* output)
{
    npConv1D(input,
             INPUT_LEN,
             conv2_partial_conv_weight,
             NULL,
             PARTIAL_CH,
             PARTIAL_CH,
             3,
             1,
             output);

    for (int c = 0; c < UNTOUCHED_CH; c++) {
        memcpy(partial_in + c * PARTIAL_LEN,
               input + (c + PARTIAL_CH) * INPUT_LEN,
               PARTIAL_LEN * sizeof(float));
    }

    npConv1D(partial_in,
             PARTIAL_LEN,
             conv2_channel_conv_weight,
             NULL,
             UNTOUCHED_CH,
             UNTOUCHED_CH,
             3,
             1,
             partial_out);

    for (int c = 0; c < UNTOUCHED_CH; c++) {
        memcpy(output + (c + PARTIAL_CH) * INPUT_LEN,
               partial_out + c * PARTIAL_LEN,
               PARTIAL_LEN * sizeof(float));

        memcpy(output + (c + PARTIAL_CH) * INPUT_LEN + PARTIAL_LEN,
               input + (c + PARTIAL_CH) * INPUT_LEN + PARTIAL_LEN,
               PARTIAL_REMAIN_LEN * sizeof(float));
    }
}

float* np_nn(float* signal)
{
    npConv1D(signal,
             INPUT_LEN,
             conv1_weight,
             conv1_bias,
             1,
             CONV1_OUT_CH,
             3,
             1,
             buf_conv1);

    Partial_conv1D(buf_conv1, buf_conv2);

    npConv1D(buf_conv2,
             INPUT_LEN,
             conv3_weight,
             conv3_bias,
             CONV1_OUT_CH,
             1,
             3,
             1,
             buf_conv3);

    npMaxPool1D(buf_conv3,
                INPUT_LEN,
                3,
                2,
                1,
                1,
                buf_pool1);

    npBatchNorm1D(buf_pool1,
                  POOL1_OUT_LEN,
                  1,
                  bn_weight,
                  bn_bias,
                  bn_running_mean,
                  bn_running_var,
                  buf_bn);

    Linear(buf_bn,
           POOL1_OUT_LEN,
           1,
           FC1_OUT,
           fc1_weight,
           fc1_bias,
           buf_fc1);

    ReLU(buf_fc1, FC1_OUT);

    Linear(buf_fc1,
           FC1_OUT,
           1,
           FC2_OUT,
           fc2_weight,
           fc2_bias,
           buf_fc2);

    return buf_fc2;
}

int findMaxIndex(const float* arr, int n)
{
    int index = 0;
    float max_val = arr[0];

    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            index = i;
        }
    }

    return index;
}

int main(void)
{
    int totalnum = 22358;
    int correct_num = 0;

    float* ECGsignal = (float*)malloc(300 * totalnum * sizeof(float));
    float* labels = (float*)malloc(totalnum * sizeof(float));
    float signal[300];

    if (ECGsignal == NULL || labels == NULL) {
        printf("memory allocation failed\n");
        return -1;
    }

    load_model_weights();

    readCSV("./X_test_np.txt", ECGsignal, 300 * totalnum);
    readLabelCSV("./Y_test_np.txt", labels, totalnum);

    for (int i = 0; i < totalnum; i++) {
        memcpy(signal, ECGsignal + i * 300, 300 * sizeof(float));

        for (int j = 0; j < 300; j++) {
            signal[j] = signal[j] / 256.0f;
        }

        float* out = np_nn(signal);
        int pred = findMaxIndex(out, FC2_OUT);

        if (pred == (int)labels[i]) {
            correct_num++;
        } else {
            printf("err_index:%d predict:%d label:%d\n", i + 1, pred, (int)labels[i]);
        }
    }

    printf("accuracy: %.2f%%\n", 100.0f * correct_num / totalnum);

    free(ECGsignal);
    free(labels);

    return 0;
}
