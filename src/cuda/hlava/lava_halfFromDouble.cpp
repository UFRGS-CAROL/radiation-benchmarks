// Created by Caio Lunardi @cblunardi the 11th Apr. 2018
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "half.hpp"


#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

#define NUMBER_PAR_PER_BOX 192

double readDoubleFromFile(FILE* file) {
    double value;
    size_t readCount = fread(&value, sizeof(double), 1, file);
    if (readCount != 1) {
        printf("Error> Failure reading from file\n");
        exit(-4);
    }
    return value;
}

half_float::half readHalfFromFile(FILE* file) {
    half_float::half value;
    size_t readCount = fread(&value, sizeof(half_float::half), 1, file);
    if (readCount != 1) {
        printf("Error> Failure reading from file\n");
        exit(-4);
    }
    return value;
}

void writeHalfToFile(FILE* file, half_float::half value) {
    size_t writeCount = fwrite(&value, sizeof(half_float::half), 1, file);
    if (writeCount != 1) {
        printf("Error> Failure writing to file\n");
        exit(-4);
    }
}

void convertValue(FILE* input, FILE* output) {
    double readValue = readDoubleFromFile(input);
    half_float::half writeValue(readValue);
    if (half_float::isinf(writeValue) || half_float::isnan(writeValue)) {
        printf("Error> Conversion failed. Input: %e ; Output: %x\n", readValue, writeValue);
        exit(-6);
    }
    writeHalfToFile(output, writeValue);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Error> Missing the number of boxes\n");
        exit(-1);
    }
    int boxes = atoi(argv[1]);
    int dimSpace = pow(boxes, 3) * NUMBER_PAR_PER_BOX;
    char inputDistancesFilename[255];
    char inputChargesFilename[255];
    char outputDistancesFilename[255];
    char outputChargesFilename[255];
    char inputDoubleGoldFilename[255];
    char inputHalfGoldFilename[255];

    snprintf(inputDistancesFilename, 254, "input_distances_double_%d", boxes);
    snprintf(inputChargesFilename, 254, "input_charges_double_%d", boxes);
    snprintf(outputDistancesFilename, 254, "input_distances_half_%d", boxes);
    snprintf(outputChargesFilename, 254, "input_charges_half_%d", boxes);
    snprintf(inputDoubleGoldFilename, 254, "output_gold_double_%d", boxes);
    snprintf(inputHalfGoldFilename, 254, "output_gold_half_%d", boxes);

    printf("Input distances (double)> %s\n", inputDistancesFilename);
    printf("Input charges (double)> %s\n", inputChargesFilename);
    printf("Output distances (half)> %s\n", outputDistancesFilename);
    printf("Output charges (half)> %s\n", outputChargesFilename);
    printf("Input gold (double)> %s\n", inputDoubleGoldFilename);
    printf("Input gold (half)> %s\n", inputHalfGoldFilename);

    std::ifstream inputExistsCheck(outputDistancesFilename);
    if (inputExistsCheck.good()) {
        // Input already exists
        printf("Half input already exists. Checking gold variation.\n");

        FILE *inputDoubleGold = fopen(inputDoubleGoldFilename, "r");
        FILE *inputHalfGold = fopen(inputHalfGoldFilename, "r");

        if (inputDoubleGold == NULL || inputHalfGold == NULL) {
            printf("Error> Failure opening input gold files\n");
            exit(-11);
        }

        float maxDiff = 0.0;
        float totDiff = 0.0;
        for (int i = 0; i < dimSpace; i++) {
            //printf(".");
            for (int dim = 0; dim < 4; dim++) {
                half_float::half halfValue = readHalfFromFile(inputHalfGold);
                double doubleValue = readDoubleFromFile(inputDoubleGold);
                float diff = abs((halfValue - doubleValue) / doubleValue);
                totDiff += diff;
                maxDiff = max(maxDiff, diff);
            }
        }
        printf("Done. Checked %d values.\nMax difference: %e\nAvg difference: %e\n", dimSpace * 4, maxDiff, totDiff / (dimSpace * 4));
    } else {
        printf("Half input does not exists. Making conversion to half.\n");

        FILE *inputDistances = fopen(inputDistancesFilename, "r");
        FILE *inputCharges = fopen(inputDistancesFilename, "r");
        FILE *outputDistances = fopen(outputDistancesFilename, "w");
        FILE *outputCharges = fopen(outputChargesFilename, "w");

        if (inputDistances == NULL || inputCharges == NULL || outputDistances == NULL || outputCharges == NULL) {
            printf("Error> Failure opening input/output distances / charges files\n");
            exit(-3);
        }

        for (int i = 0; i < dimSpace; i++) {
            //printf(".");
            for (int dim = 0; dim < 4; dim++) {
                convertValue(inputDistances, outputDistances);
            }
            convertValue(inputCharges, outputCharges);
        }
        printf("Done. Converted %d values.\n", dimSpace * 5);
    }
}