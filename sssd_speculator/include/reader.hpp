/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#ifndef READER_HPP
#define READER_HPP

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <prompt_cache.hpp>
#include <utils.hpp>

struct SubIndex {
    std::vector<int32_t> data;
    std::vector<int32_t> indexData;
    std::atomic<size_t> maxSearchEntries;
};

class Reader {
public:
    Reader(const std::string &indexFilePath, int stopToken, int maxSearchEntries, int promptBranchLength,
        int promptPrefixLength, int maxOutPutLength, bool liveCacheUpdates, int maxChunkSize, int maxIndexes,
        int updateIntervalMs, int vocabSize, int promptTokensInDatastore, std::size_t max_topk);

    ~Reader();

    std::pair<std::vector<int>, std::vector<int>> GetBatchElementCandidates(const std::vector<int> &prefix,
        int decoding_length, int branchLength, std::shared_ptr<bool[]> &mask_buffer, int seqId, Trie &full_trie);

    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<py::array_t<bool>>>
        GetCandidates(const std::vector<std::vector<int>> &prefixes, const std::vector<int> &decodingLengths,
            const std::vector<int> &branchLengths, const std::vector<int> &seqIds);

    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> GetBatchElementCandidatesSglang(
        const std::vector<int> &prefix, int decoding_length, int branchLength, std::shared_ptr<bool[]> &mask_buffer,
        int seqId, Trie &full_trie);
    
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<std::vector<int>>,
        std::vector<std::vector<int>>, std::vector<py::array_t<bool>>> GetCandidatesSglang(
            const std::vector<std::vector<int>> &prefixes, const std::vector<int> &decodingLengths,
            const std::vector<int> &branchLengths, const std::vector<int> &seqIds);

    void Put(std::vector<int> &input, int seqId);

    void AsyncPut(std::vector<int> &&input, int seqId);

    void StreamPut(std::vector<int> &newTokens, int seqId);

    void BatchedStreamPut(const std::vector<std::pair<std::vector<int>, int>> &newTokens);

    void FinishSequence(int seqId);

    void AsyncFinishSequence(int seqId);

    void FinishAll();

    void SaveIndexesToDisk(const std::string &path);

    void UpdateAttributes(
        int stopToken, int maxSearchEntries, int promptBranchLength, int promptPrefixLength,
        int maxOutPutSize, int inputTokensToPutInSelfOutput);

    void SetUpdatePauseFlag(bool flag);

    void PrintIndexes();

private:
    bool hasDatastore;
    std::deque<std::shared_ptr<SubIndex>> indexes;
    PromptCache promptCache;
    int32_t stopToken;
    int maxOutPutSize;
    int maxSearchEntries;
    std::shared_ptr<TrieNodeBlockPool> nodeBlockPool;
    std::shared_ptr<VectorPool> selfOutputPool;
    // Just for memory pooling, to avoid destroyng the trie once it's created
    std::vector<Trie> datastoreTriesToBuild;
    // For SGLang
    std::vector<std::vector<int>> firstChildren;
    std::vector<std::vector<int>> nextSiblings;
    std::vector<std::future<std::tuple<std::vector<int>, std::vector<int>,
            std::vector<int>, std::vector<int>>>> sglangFutures;
    std::size_t max_topk;

    // Object for efficient candidates retrieval
    std::shared_ptr<ThreadPool> threadPool;
    std::vector<std::vector<int>> candidates;
    std::vector<std::vector<int>> depths;
    std::vector<std::shared_ptr<uint16_t[]>> rawMasks;
    std::vector<py::array_t<uint16_t>> masks;
    std::vector<std::shared_ptr<bool[]>> rawMasksBool;
    std::vector<py::array_t<bool>> boolMasks;
    std::vector<std::future<std::pair<std::vector<int>, std::vector<int>>>> futures;

    // For live cache updates (on separate thread)
    bool updateDatastore;
    std::deque<std::vector<int>> newSentences;
    std::mutex newSentencesMtx;
    std::mutex indexesMtx;
    std::thread updateThread;
    std::atomic<bool> stopThread;
    int64_t updateIntervalMs;
    size_t maxChunkSize;
    int vocabSize;
    int maxIndexes;
    std::atomic<bool> datastoreUpdatePause;
    // Use this to recycle memory allocations and avoid object destructions
    std::shared_ptr<SubIndex> newSubindex;

    std::shared_ptr<spdlog::logger> logger;

    void SearchCandidates(const std::vector<int32_t> &prefix,
        size_t branchLength,  // how many tokens per suffix to get to build the trie from each subindex
        Trie &trie);

    void UpdateIndexes();

    void WriteIndexesToDisk(const std::string &path);
};

// UTILS

enum class CompareResult { LESS, EQUAL, GREATER };

CompareResult compare_ranges(std::vector<int32_t>::const_iterator start1, std::vector<int32_t>::const_iterator end1,
    std::vector<int32_t>::const_iterator start2, std::vector<int32_t>::const_iterator end2);

inline std::string print_indexes(const std::deque<std::shared_ptr<SubIndex>> &indexes)
{
    std::ostringstream logMessage;
    logMessage << " Indices: ";
    for (const auto &ptr : indexes) {
        logMessage << " " << ptr.get();
    }
    return logMessage.str();
}

#endif  // READER_HPP