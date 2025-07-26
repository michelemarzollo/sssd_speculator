/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <writer.hpp>

namespace py = pybind11;

Writer::Writer(const std::string &indexFilePath, int vocabSize, size_t maxChunkSize) : maxChunkSize(maxChunkSize)
{
    this->vocabSize = vocabSize;
    this->buffer.reserve(maxChunkSize);

    // Check if file exists and decide the action
    logger = GetLogger();

    if (FileExists(indexFilePath)) {
        if (logger != nullptr) {
            SPDLOG_LOGGER_INFO(logger, "The file exists! The new data will be added to the already present data.");
        } else {
            std::cout << "The file exists! The new data will be added to the already present data." << std::endl;
        }
        try {
            std::tie(indexFile, buffer) = ResumeFromLastChunk(indexFilePath, maxChunkSize);
            // Use out_file and buffer as needed
            // If you need to check something specific about the results, do it here
            if (buffer.empty()) {
                SPDLOG_LOGGER_ERROR(logger, "Warning: Buffer is empty after resuming.");
            }
        } catch (const std::exception &e) {
            std::ostringstream oss;
            oss << "Failed to resume from last chunk: " << e.what();
            throw py::value_error(oss.str());
        }
    } else {
        // File does not exist, create new
        this->indexFile.open(indexFilePath, std::ios::out | std::ios::binary);
        if (!this->indexFile.is_open()) {
            throw std::runtime_error("Failed to create the file.");
        }

        if (vocabSize > 65536) {
            // Write a flag for the reader on how to read data
            int32_t flag = 0;
            this->indexFile.write(reinterpret_cast<char *>(&flag), sizeof(flag));
        }
        this->indexFile.close();
        this->indexFile.open(indexFilePath, std::ios::in | std::ios::out | std::ios::binary);
        if (this->indexFile.is_open()) {
            this->indexFile.seekp(0, std::ios::end);  // Append from the end
        }
    }
}

std::pair<std::ofstream, std::vector<int32_t>> Writer::ResumeFromLastChunk(
    const std::string &indexFilePath, size_t maxChunkSize)
{
    std::ifstream indexFile(indexFilePath, std::ios::binary);
    if (!indexFile.is_open()) {
        throw std::runtime_error("Failed to open the file for reading.");
    }

    // Getting file length
    indexFile.seekg(0, std::ios::end);
    size_t indexFile_len = indexFile.tellg();
    indexFile.seekg(0, std::ios::beg);

    std::vector<int32_t> buffer;
    buffer.reserve(maxChunkSize);

    size_t bytes_read = 0;
    size_t append_from = indexFile_len;

    int32_t first_flag;
    indexFile.read(reinterpret_cast<char *>(&first_flag), sizeof(first_flag));
    bytes_read += sizeof(first_flag);

    size_t token_size = first_flag == 0 ? 4 : 2;

    if (first_flag != 0) {
        indexFile.seekg(0, std::ios::beg);
        bytes_read = 0;
    }

    while (bytes_read < indexFile_len) {
        size_t last_chunk_start = bytes_read;

        // Go through the chunk text data
        uint32_t data_file_len;
        indexFile.read(reinterpret_cast<char *>(&data_file_len), sizeof(data_file_len));
        indexFile.seekg(data_file_len, std::ios::cur);

        // Go through the chunk index data
        uint32_t suffixes_file_len;
        indexFile.read(reinterpret_cast<char *>(&suffixes_file_len), sizeof(suffixes_file_len));
        indexFile.seekg(suffixes_file_len, std::ios::cur);

        bytes_read += sizeof(data_file_len) + sizeof(suffixes_file_len) + data_file_len + suffixes_file_len;
        if (bytes_read + sizeof(int32_t) > indexFile_len) {
            // It's the last chunk!
            // Move the cursor to the beginning of the data chunk, to read it and append to it
            indexFile.seekg(last_chunk_start + sizeof(data_file_len), std::ios::beg);

            std::vector<char> data_u8(data_file_len);
            indexFile.read(data_u8.data(), data_file_len);

            if (data_u8.size() / token_size < maxChunkSize) {
                std::vector<int32_t> last_chunk_text_data;
                if (token_size == 4) {
                    for (size_t i = 0; i + 3 < data_u8.size(); i += 4) {
                        int32_t value;
                        std::copy(&data_u8[i], &data_u8[i] + sizeof(int32_t), reinterpret_cast<char *>(&value));
                        last_chunk_text_data.push_back(value);
                    }
                } else if (token_size == 2) {
                    for (size_t i = 0; i + 1 < data_u8.size(); i += 2) {
                        int16_t short_value;
                        std::copy(&data_u8[i], &data_u8[i] + sizeof(int16_t), reinterpret_cast<char *>(&short_value));
                        last_chunk_text_data.push_back(static_cast<int>(short_value));
                    }
                }
                buffer.insert(buffer.end(), last_chunk_text_data.begin(), last_chunk_text_data.end());
                append_from = last_chunk_start;
            }
            // else, all the chunks are complete (very unlikely), just append from the end
            // (use default 'buffer' and 'append_from')
        }
    }

    // Prepare file for appending
    std::ofstream out_file(indexFilePath, std::ios::binary | std::ios::out | std::ios::in | std::ios::ate);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open the file for writing.");
    }

    out_file.seekp(append_from, std::ios::beg);

    return {std::move(out_file), buffer};
}

void Writer::AddEntry(const std::vector<int> &tokenizedSentence)
{
    if (tokenizedSentence.size() > maxChunkSize) {
        throw py::value_error("Entry is to big, if you want to insert it, split it into pieces of at most " +
                              std::to_string(maxChunkSize) + " tokens. The list you tried to insert was of length " +
                              std::to_string(tokenizedSentence.size()) + ".");
    }
    if (this->buffer.size() + tokenizedSentence.size() > maxChunkSize) {
        DumpData();
    }

    this->buffer.insert(this->buffer.end(), tokenizedSentence.begin(), tokenizedSentence.end());
}

void Writer::DumpData()
{
    if (buffer.empty())
        return;

    uint32_t data_size = buffer.size() * (vocabSize > 65536 ? 4 : 2);
    indexFile.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));

    if (vocabSize > 65536) {
        // Write the entire buffer at once
        indexFile.write(reinterpret_cast<const char *>(buffer.data()), buffer.size() * sizeof(int32_t));
    } else {
        // Convert buffer to uint16_t and write all at once
        std::vector<uint16_t> shortBuffer(buffer.size());
        std::transform(buffer.begin(), buffer.end(), shortBuffer.begin(), [](int32_t value) {
            return static_cast<uint16_t>(value);
        });
        indexFile.write(reinterpret_cast<const char *>(shortBuffer.data()), shortBuffer.size() * sizeof(uint16_t));
    }

    auto logger = spdlog::get("sssd_speculator");
    if (logger != nullptr) {
        SPDLOG_LOGGER_INFO(logger, "Constructing suffix array");
    } else {
        std::cout << "Constructing suffix array" << std::endl;
    }
    auto suffix_array = constructSuffixArray(buffer, vocabSize, logger);
    if (logger != nullptr) {
        SPDLOG_LOGGER_INFO(logger, "Finished constructing suffix array");
    } else {
        std::cout << "Finished constructing suffix array" << std::endl;
    }

    uint32_t suffix_array_size = suffix_array.size() * 4;
    indexFile.write(reinterpret_cast<const char *>(&suffix_array_size), sizeof(suffix_array_size));
    for (int32_t suffix : suffix_array) {
        indexFile.write(reinterpret_cast<const char *>(&suffix), sizeof(suffix));
    }

    buffer.clear();
    buffer.reserve(maxChunkSize);
}

void Writer::Finalize()
{
    if (!this->buffer.empty()) {
        DumpData();
    }

    // Flush the file to ensure all data is written
    this->indexFile.flush();
    if (!this->indexFile.good()) {
        std::string error_message = "Failed to flush the file correctly. ";
        if (this->indexFile.bad()) {
            error_message += "Reason: Irrecoverable stream error (bad bit set).";
        } else if (this->indexFile.fail()) {
            error_message += "Reason: Logical error on i/o operation (fail bit set).";
        } else if (this->indexFile.eof()) {
            error_message += "Reason: End-of-File reached (eof bit set).";
        }
        throw std::runtime_error(error_message);
    }
}

Writer::~Writer()
{
    try {
        Finalize();
    } catch (const std::exception &e) {
        // No problem
    }
    if (this->indexFile.is_open()) {
        this->indexFile.close();
    }
}
