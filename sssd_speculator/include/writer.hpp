/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#ifndef WRITER_HPP
#define WRITER_HPP

#include <fstream>
#include <vector>
#include <iostream>
#include <utils.hpp>

class Writer {
public:
    Writer(const std::string &indexFilePath, int vocabSize, size_t maxChunkSize = 512 * 1024 * 1024);
    void AddEntry(const std::vector<int> &pyText);
    void Finalize();
    ~Writer();

private:
    std::ofstream indexFile;
    std::vector<int32_t> buffer;
    int vocabSize;
    size_t maxChunkSize;
    std::shared_ptr<spdlog::logger> logger;

    std::pair<std::ofstream, std::vector<int32_t>> ResumeFromLastChunk(
        const std::string &indexFilePath, size_t maxChunkSize);
    void DumpData();
};

#endif  // WRITER_HPP