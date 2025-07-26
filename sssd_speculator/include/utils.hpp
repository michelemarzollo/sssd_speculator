/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#ifndef UTILS_HPP
#define UTILS_HPP

#if !defined(SPDLOG_ACTIVE_LEVEL)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN
#endif

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <iostream>
#include <thread>
#include <sched.h>
#include <bitset>
#include <unistd.h>
#include <sstream>
#include <algorithm>
#include <libsais.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

constexpr int PRIORITY_HIGH = 1;
constexpr int PRIORITY_LOW = 0;

// Check system endianness
inline bool IsLittleEndian()
{
    uint16_t number = 0x1;
    return *(reinterpret_cast<uint8_t *>(&number)) == 0x1;
}

inline bool FileExists(const std::string &name)
{
    std::ifstream file(name.c_str());
    return file.good();
}

// Function to get the logger, creates it if it doesn't exist.
std::shared_ptr<spdlog::logger> GetLogger();

// Adapter to libsais
inline std::vector<int32_t> constructSuffixArray(
    const std::vector<int32_t> &buffer, int32_t vocabSize, std::shared_ptr<spdlog::logger> logger)
{
    if (buffer.empty()) {
        SPDLOG_LOGGER_WARN(logger, "Empty data, not constructing the suffix array");
        return std::vector<int32_t>();
    }
    std::vector<int32_t> modifiableBuffer = buffer;
    std::vector<int32_t> suffixArray(modifiableBuffer.size(), 0);
    auto max_value = *std::max_element(modifiableBuffer.begin(), modifiableBuffer.end());
    auto min_value = *std::min_element(modifiableBuffer.begin(), modifiableBuffer.end());

    if (min_value < 0 || max_value >= vocabSize) {
        SPDLOG_LOGGER_ERROR(logger,
            "Error: Buffer contains invalid values. min: {}, max: {}, vocab size {}",
            min_value,
            max_value,
            vocabSize);
        return std::vector<int32_t>();
    }
    int returnValue = libsais_int(
        const_cast<int32_t *>(modifiableBuffer.data()), suffixArray.data(), modifiableBuffer.size(), vocabSize, 0);
    if (returnValue != 0) {
        SPDLOG_LOGGER_ERROR(logger, "libsais_int returned value {}. Suffix array was not constructed.", returnValue);
        return std::vector<int32_t>();
    }
    return suffixArray;
}

inline int constructSuffixArrayInplace(std::vector<int32_t> &dataBuffer, std::vector<int32_t> &suffixArray,
    int32_t vocabSize, std::shared_ptr<spdlog::logger> logger)
{
    if (dataBuffer.empty()) {
        SPDLOG_LOGGER_WARN(logger, "Empty data, not constructing the suffix array");
        return -1;
    }
    suffixArray.resize(dataBuffer.size(), 0);

    auto max_value = *std::max_element(dataBuffer.begin(), dataBuffer.end());
    auto min_value = *std::min_element(dataBuffer.begin(), dataBuffer.end());

    if (min_value < 0 || max_value >= vocabSize) {
        SPDLOG_LOGGER_ERROR(logger,
            "Error: Buffer contains invalid values. min: {}, max: {}, vocab size {}",
            min_value,
            max_value,
            vocabSize);
        return -1;
    }

    std::vector<int32_t> backupData = dataBuffer;

    int returnValue = libsais_int(dataBuffer.data(), suffixArray.data(), dataBuffer.size(), vocabSize, 0);
    if (returnValue != 0) {
        // The original data vector might have been modified
        dataBuffer = backupData;
        SPDLOG_LOGGER_ERROR(logger, "libsais_int returned value {}. Suffix array was not constructed.", returnValue);
    }
    return returnValue;
}

inline size_t GetPinnedCpuCount()
{
    // Get the number of CPU cores in the system
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // Get the current process's CPU affinity
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
        std::cerr << "Error getting CPU affinity" << std::endl;
        return 0;
    }

    // Count the number of CPUs that are part of the affinity mask
    size_t count = 0;
    std::ostringstream cpuStream;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpuset)) {
            ++count;
            if (cpuStream.tellp() > 0) {
                cpuStream << ", ";
            }
            cpuStream << i;
        }
    }
    auto logger = GetLogger();
    SPDLOG_LOGGER_DEBUG(logger, "Pinned CPUs: {}", cpuStream.str());
    return count;
}

inline std::vector<int> get_available_cores()
{
    std::vector<int> available_cores;
    const char *tasksetOutput = "taskset -c $(taskset -p $$ | sed 's/^[^:]*: //')";
    FILE *fp = popen(tasksetOutput, "r");

    if (fp == nullptr) {
        std::cerr << "Failed to get available cores" << std::endl;
        return available_cores;
    }

    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), fp)) {
        result += buffer;
    }
    fclose(fp);

    // Parse the available cores from the taskset output
    std::istringstream iss(result);
    std::string token;
    while (std::getline(iss, token, ',')) {
        available_cores.push_back(std::stoi(token));
    }

    return available_cores;
}

inline void PinThreadToCore(int coreId)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreId, &cpuset);  // Pin to the core_id

    pthread_t currentThread = pthread_self();
    int ret = pthread_setaffinity_np(currentThread, sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        std::cerr << "Failed to set thread affinity: " << ret << std::endl;
    } else {
        std::cout << "Thread pinned to core " << coreId << std::endl;
    }
}

#endif  // UTILS_HPP