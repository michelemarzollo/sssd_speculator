/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include <prompt_cache.hpp>
#include <utils.hpp>

// Sequence CODE

Sequence::Sequence(int branchLength, int prefixLength, int inputTokensToPutInSelfOutput,
    std::shared_ptr<ThreadPool> pool, std::shared_ptr<TrieNodeBlockPool> trieBlockPool,
    std::shared_ptr<VectorPool> vectorPool)
    : branchLength(branchLength), prefixLength(prefixLength), inputTokensToPutInSelfOutput(inputTokensToPutInSelfOutput),
      nextToInsert(0), cache(-1, *trieBlockPool), putStarted(false), putFinished(false), stopPutThread(false), threadPool(pool), vectorPool(vectorPool),
      selfOutputInvalid(false)
{
    selfOutput = vectorPool->Acquire();  // reserve more in case there are special tokens added after end
    logger = GetLogger();
}

Sequence::~Sequence()
{
    stopPutThread = true;
    if (!selfOutputInvalid.load()) {
        // If the output was not picked up by someone else for processing, put it back
        // to the vector pool.
        vectorPool->Release(std::move(selfOutput));
    }
}

/* Clear the Sequence for new use. */
void Sequence::Clear(bool dontClearTreeIfNeeded)
{
    nextToInsert = 0;
    putStarted.store(false);
    putFinished.store(false);
    stopPutThread.store(false);
    if (selfOutputInvalid.load()) {
        // If the output was moved, substitute the vector
        if (selfOutput.size() != 0) {
            SPDLOG_LOGGER_DEBUG(logger, "selfOutput was not moved: size: {}", selfOutput.size());
        }
        selfOutput = vectorPool->Acquire();
    } else {
        // No need to go through the vector pool, just clear the vector for reuse
        selfOutput.clear();
    }
    selfOutputInvalid.store(false);
    cache.Clear(dontClearTreeIfNeeded);
}

void Sequence::Put(std::vector<int> &input)
{
    putStarted.store(true);

    // // Last few tokens can be attached to self output to connect end of prompt to response
    // const std::size_t reserve = std::min<std::size_t>(inputTokensToPutInSelfOutput, input.size());
    // if (reserve) {
    //     selfOutput.insert(selfOutput.end(), input.end() - reserve, input.end());
    // }

    auto start_iter = input.begin();
    auto max_iter = input.end();
    while (start_iter < max_iter - 1) { //inputTokensToPutInSelfOutput) {  // Don't insert single tokens
        cache.InsertIter(start_iter, std::min(start_iter + branchLength, max_iter));
        start_iter++;
    }
    putFinished.store(true);
}

void Sequence::AsyncPut(std::vector<int> &&input)
{
    // Not meant to be called concurrently to StreamPut. Concurrency starts here.
    // Still, it could be called later than StreamPut.
    if (putStarted.load() && !putFinished.load()) {
        // Silent return to avoid breaking, it should almost never happen
        return;
    }
    putStarted.store(true);
    putFinished.store(false);
    stopPutThread = false;  // in case this is not the first AsyncPut call

    // const std::size_t reserve = std::min<std::size_t>(inputTokensToPutInSelfOutput, input.size());
    // if (reserve) {
    //     selfOutput.insert(selfOutput.end(), input.end() - reserve, input.end());
    // }

    auto inputPointer = std::make_shared<std::vector<int>>(input);

    // Capture a shared_ptr to this to keep the Sequence alive during the async operation
    auto self = shared_from_this();

    // Enqueue the PutThreadFunction to the ThreadPool
    threadPool->Enqueue(PRIORITY_LOW, [self, inputPointer]() { self->PutThreadFunction(inputPointer); });
}

void Sequence::PutThreadFunction(std::shared_ptr<std::vector<int>> input)
{
    // Don't modify selfOutput: it's not thread safe
    auto start_iter = input->begin();
    auto max_iter = input->end();
    while (start_iter < max_iter - 1 && !stopPutThread.load()) {  // Don't insert single tokens
        cache.InsertIter(start_iter, std::min(start_iter + branchLength, max_iter));
        start_iter++;
    }
    putFinished = true;
}

void Sequence::StreamPut(const std::vector<int> &new_tokens)
{
    /* Add tokens generated in the last forward pass */
    if (!putStarted.load()) {
        // If put was not called yet, pretend it was called
        putStarted = true;
        putFinished = true;
    }
    selfOutput.insert(selfOutput.end(), new_tokens.begin(), new_tokens.end());
    if (putFinished.load()) {
        auto nextToInsertIter = selfOutput.begin() + nextToInsert;
        while (selfOutput.end() - nextToInsertIter >= branchLength) {
            cache.InsertIter(nextToInsertIter, nextToInsertIter + branchLength);
            nextToInsertIter++;
        }
        nextToInsert = nextToInsertIter - selfOutput.begin();
    }
    // If the Put thread is not finished yet, don't update the cache tree in this call
}

std::vector<std::pair<TrieNode *, int>> Sequence::FindPrefix(const std::vector<int> &prefix)
{
    if (!putFinished.load()) {
        return {};  // Return empty if Put hasn't finished
    }
    // Returns all prefixes of size [prefix.size(), prefix.size()-1, ..., 1] that can be found in
    // the input + self output. Each element contains the node from which to pick the continuations, and the
    // size of the matched prefix.
    std::vector<std::pair<TrieNode *, int>> results;
    size_t iter = 0;
    if (prefix.size() >= this->prefixLength) {
        // for the input we usually want to match a shorter prefix (prefixLength of cache is usually smaller?)
        iter = prefix.size() - prefixLength;
    }
    while (iter < prefix.size()) {
        size_t matched_idx = iter;
        auto current_node = cache.GetRoot();
        while (matched_idx < prefix.size() && current_node->children.count(prefix[matched_idx])) {
            current_node = current_node->children[prefix[matched_idx]];
            matched_idx++;
        }
        if (matched_idx == prefix.size()) {
            results.emplace_back(current_node, prefix.size() - iter);
        }
        iter++;
    }
    return results;
}

std::vector<int> Sequence::GetSelfOutputOwnership()
{
    // Not meant to be called concurrently, it's not really thread safe. It actually should be
    // called only once in practice.
    if (selfOutputInvalid.load()) {
        return std::vector<int>();
    }
    selfOutputInvalid.store(true);
    return std::move(selfOutput);
}

// PromptCache CODE

PromptCache::PromptCache(int branchLength, int prefixLength, int inputTokensToPutInSelfOutput)
    : branchLength(branchLength), prefixLength(prefixLength),
      inputTokensToPutInSelfOutput(inputTokensToPutInSelfOutput), sequencePool()
{
    asyncMemoryFree = std::thread(&PromptCache::AsyncFinishWorker, this);
    logger = GetLogger();
}

PromptCache::~PromptCache()
{
    running = false;
    if (asyncMemoryFree.joinable()) {
        asyncMemoryFree.join();
    }
}

void PromptCache::SetPools(std::shared_ptr<ThreadPool> threadPool, std::shared_ptr<TrieNodeBlockPool> trieBlockPool,
    std::shared_ptr<VectorPool> vectorPool)
{
    this->threadPool = threadPool;
    this->trieBlockPool = trieBlockPool;
    this->vectorPool = vectorPool;
    this->sequencePool =
        SequencePool(branchLength, prefixLength, inputTokensToPutInSelfOutput, this->threadPool, this->trieBlockPool, this->vectorPool);
}

void PromptCache::Put(std::vector<int> &input, int seqId)
{
    auto it = sequences.find(seqId);
    if (it != sequences.end()) {
        it->second->Put(input);
    } else {
        // Use pointer, otherwise you need to deal with copies of raw pointers
        auto res = sequences.emplace(seqId, sequencePool.Acquire());
        res.first->second->Put(input);
    }
    SPDLOG_LOGGER_TRACE(logger, "Total sequences: {}", sequences.size());
}

void PromptCache::AsyncPut(std::vector<int> &&input, int seqId)
{
    auto it = sequences.find(seqId);
    if (it != sequences.end()) {
        it->second->AsyncPut(std::move(input));
    } else {
        // Use pointer, otherwise you need to deal with copies of raw pointers
        auto res = sequences.emplace(seqId, sequencePool.Acquire());
        res.first->second->AsyncPut(std::move(input));
    }
    SPDLOG_LOGGER_TRACE(logger, "Total sequences: {}", sequences.size());
}

void PromptCache::StreamPut(const std::vector<int> &new_tokens, int seqId)
{
    auto it = sequences.find(seqId);
    if (it != sequences.end()) {
        it->second->StreamPut(new_tokens);
    } else {
        // Edge case: Insert and then access (should not happen that there is no prompt)
        auto res = sequences.emplace(seqId, sequencePool.Acquire());
        res.first->second->StreamPut(new_tokens);
    }
}

void PromptCache::FinishSequence(int seqId)
{
    auto it = sequences.find(seqId);
    if (it != sequences.end()) {
        it->second->stopPutThread = true;
        sequencePool.Release(it->second, false);
        sequences.erase(it);
    }
}

void PromptCache::AsyncFinishSequence(int seqId)
{
    auto it = sequences.find(seqId);
    if (it != sequences.end()) {
        it->second->stopPutThread = true;
        {
            // Move to the queue to defer destruction
            std::lock_guard<std::mutex> lock(mtx);
            finishedSequencesQueue.push_back(it->second);
        }
        sequences.erase(it);
    }
}

std::vector<int32_t> PromptCache::FinishSequenceAndGetOutput(int seqId)
{
    std::vector<int32_t> sentence;

    auto it = sequences.find(seqId);
    if (it != sequences.end()) {
        // In case Put is running asynchronously, make sure it terminated
        auto sequence = it->second;
        sequence->stopPutThread = true;  // In case Put is still running asynchronously
        sentence = sequence->GetSelfOutputOwnership();
        sequencePool.Release(sequence, false);
        sequences.erase(it);
    }
    return sentence;
}

std::vector<int32_t> PromptCache::AsyncFinishSequenceAndGetOutput(int seqId)
{
    std::vector<int32_t> sentence;

    auto it = sequences.find(seqId);
    if (it != sequences.end()) {
        // In case Put is running asynchronously, make sure it terminated
        auto sequence = it->second;
        sequence->stopPutThread = true;  // In case Put is still running asynchronously
        sentence = sequence->GetSelfOutputOwnership();
        {
            // Move to the queue to defer destruction
            std::lock_guard<std::mutex> lock(mtx);
            finishedSequencesQueue.push_back(sequence);
        }
        sequences.erase(it);
    }
    return sentence;
}

void PromptCache::FinishAll()
{
    // Use a copy of sequence IDs to avoid modifying the container while iterating
    std::vector<int> sequenceIds;
    for (const auto &entry : sequences) {
        sequenceIds.push_back(entry.first);
    }

    for (int seqId : sequenceIds) {
        FinishSequence(seqId);
    }
}

void PromptCache::SetParameters(int branchLength, int prefixLength, int inputTokensToPutInSelfOutput)
{
    branchLength = branchLength;
    prefixLength = prefixLength;
    inputTokensToPutInSelfOutput = inputTokensToPutInSelfOutput;
}

void PromptCache::AsyncFinishWorker()
{
    std::vector<std::shared_ptr<Sequence>> localFinishData;
    while (running) {
        while (!finishedSequencesQueue.empty()) {
            // Move data from queues to local variables
            {
                std::lock_guard<std::mutex> lock(mtx);
                while (!finishedSequencesQueue.empty()) {
                    localFinishData.push_back(finishedSequencesQueue.front());
                    finishedSequencesQueue.pop_front();
                }
            }
            for (const auto &sequence : localFinishData) {
                // the second parameter is for backpressure handling: if the sequences are queuing, it
                // might make sense to not clear the content before returning it to the pool (also need
                // to check if the pool is indeed struggling to return sequences
                sequencePool.Release(sequence, (localFinishData.size() > 2));
            }
            localFinishData.clear();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}