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

size_t GetPinnedCpuCount();
void PinThreadToCore(int coreId);

// NUMA restrictions

enum class AffinityScope {
    kAny,          // No extra restriction: keep current affinity as-is
    kLocalNode,    // Restrict to NUMA node of current CPU if available; else no-op
    kSpecificCore  // Pin to a specific core (validate it is within local node if NUMA)
};


int GetCurrentNUMANode(std::shared_ptr<spdlog::logger> logger);
bool BuildCpusetForNode(int node, cpu_set_t &out, std::shared_ptr<spdlog::logger> logger);
bool PinThreadToLocalNUMANode(std::shared_ptr<spdlog::logger> logger);
void PreferLocalNodeMemory(std::shared_ptr<spdlog::logger> logger);
bool ConstrainProcessToLocalNode(std::shared_ptr<spdlog::logger> logger);
// Apply the cpuset to the current thread
bool ApplyThreadAffinity(const cpu_set_t& set, std::shared_ptr<spdlog::logger> logger);

// Builds a cpuset according to the requested scope.
// - For kAny: just returns current affinity.
// - For kLocalNode: intersect(current_affinity, local_node_cpus) if NUMA available; else current_affinity.
// - For kSpecificCore: returns only that core; if NUMA available, also verifies it’s in the local node.
bool BuildEffectiveCpuset(
    AffinityScope scope,
    cpu_set_t& out,
    int core_id /* used only for kSpecificCore */,
    std::shared_ptr<spdlog::logger> logger);

std::vector<int> GetEffectiveCores(AffinityScope scope, std::shared_ptr<spdlog::logger> logger, int core_id = -1);

#endif  // UTILS_HPP