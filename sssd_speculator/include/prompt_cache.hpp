/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#ifndef PROMPT_CACHE_HPP
#define PROMPT_CACHE_HPP

#include <unordered_map>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <deque>
#include <ThreadPool.h>
#include <trie.hpp>

/*
Note that the methods in this file are not meant to be run asynchronously. Async versions are
provided (methods starting with Async), but if you are calling methods in this (and other files)
you should implement proper synchronization.
*/
class Sequence : public std::enable_shared_from_this<Sequence> {
public:
    Trie cache;
    // For async put (useful for very large prompt), it's kind of a lightweight lock
    std::atomic<bool> putStarted{false};
    std::atomic<bool> putFinished{false};
    std::atomic<bool> stopPutThread{false};

    Sequence(int branchLength, int prefixLength, int inputTokensToPutInSelfOutput, std::shared_ptr<ThreadPool> pool,
        std::shared_ptr<TrieNodeBlockPool> trieBlockPool, std::shared_ptr<VectorPool> vectorPool);
    ~Sequence();
    void Put(std::vector<int> &input);
    void StreamPut(const std::vector<int> &new_tokens);
    void AsyncPut(std::vector<int> &&input);
    // returns the node in the trie from which you should get the continuations and how long the matched prefix was
    std::vector<std::pair<TrieNode *, int>> FindPrefix(const std::vector<int> &prefix);
    std::vector<int> GetSelfOutputOwnership();  // Not meant to be called concurrently
    void Clear(bool dontClearTreeIfNeeded);

private:
    int branchLength;
    size_t prefixLength;
    size_t nextToInsert;
    std::vector<int> selfOutput;
    std::atomic<bool> selfOutputInvalid;
    std::shared_ptr<ThreadPool> threadPool;
    std::shared_ptr<VectorPool> vectorPool;
    std::shared_ptr<spdlog::logger> logger;
    // Tokens from the input that you want to keep with the self-output to improve the tree coherence and to
    // save some more context in the live datastore.
    size_t inputTokensToPutInSelfOutput;

    void PutThreadFunction(std::shared_ptr<std::vector<int>> input);
};

/*
    Pool to recycle Sequences. Implemented in memory_pool.cpp
*/
class SequencePool {
public:
    SequencePool();
    SequencePool(int branchLength, int prefixLength, int inputTokensToPutInSelfOutput, std::shared_ptr<ThreadPool> threadPool,
        std::shared_ptr<TrieNodeBlockPool> trieBlockPool, std::shared_ptr<VectorPool> vectorPool);
    std::shared_ptr<Sequence> Acquire();
    void Release(std::shared_ptr<Sequence> seq, bool multipleWaiting);
    SequencePool &operator=(SequencePool &&other) noexcept;

private:
    std::mutex poolMutex;
    std::deque<std::shared_ptr<Sequence>> sequencePool;
    int branchLength;
    int prefixLength;
    int inputTokensToPutInSelfOutput;
    size_t maxOutputSize;
    std::shared_ptr<ThreadPool> threadPool;
    std::shared_ptr<TrieNodeBlockPool> trieBlockPool;
    std::shared_ptr<VectorPool> vectorPool;
};

class PromptCache {
public:
    std::unordered_map<int, std::shared_ptr<Sequence>> sequences;

    PromptCache(int branchLength, int prefixLength, int inputTokensToPutInSelfOutput);
    ~PromptCache();
    void SetPools(std::shared_ptr<ThreadPool> threadPool, std::shared_ptr<TrieNodeBlockPool> trieBlockPool,
        std::shared_ptr<VectorPool> vectorPool);
    void Put(std::vector<int> &input, int seqId);
    void StreamPut(const std::vector<int> &new_tokens, int seqId);
    void AsyncPut(std::vector<int> &&input, int seqId);
    void FinishSequence(int seqId);
    void AsyncFinishSequence(int seqId);
    std::vector<int32_t> FinishSequenceAndGetOutput(int seqId);
    std::vector<int32_t> AsyncFinishSequenceAndGetOutput(int seqId);
    void FinishAll();
    void SetParameters(int branchLength, int prefixLength, int inputTokensToPutInSelfOutput);
    void AsyncFinishWorker();

private:
    int branchLength;
    size_t prefixLength;
    int inputTokensToPutInSelfOutput;

    // For sequences creation
    std::shared_ptr<ThreadPool> threadPool;

    std::shared_ptr<TrieNodeBlockPool> trieBlockPool;
    std::shared_ptr<VectorPool> vectorPool;
    SequencePool sequencePool;

    // For async finish
    std::mutex mtx;
    std::deque<std::shared_ptr<Sequence>> finishedSequencesQueue;
    std::thread asyncMemoryFree;
    std::atomic<bool> running;

    std::shared_ptr<spdlog::logger> logger;
};

#endif  // PROMPT_CACHE_HPP