/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include <vector>
#include <queue>
#include <mutex>
#include <trie.hpp>
#include <prompt_cache.hpp>

/* Just some additional safety in case, for example, of special tokens added to the model output
(probably not needed). Just to make sure we don't deallocate and reallocate. */
static const int ADDITIONAL_CAPACITY = 10;

/* When clearing a block of TrieNodes, to cap the memory usage, we avoid keeping in the cleared nodes
very large hashmaps. If there were more than these buckets in the children hashmap, it will be deallocated
and changed with a new hashmap */
static const int MAX_CHILDREN_OF_EMPTY_NODE = 64;

/* The nodes in the first block, the one that will be recycled, will on average have more children (more
likely to be close to root, at least some of them): let's be less strict with deallocation, this are just
a few blocks. */
static const int MAX_CHILDREN_OF_EMPTY_NODE_FIRST_BLOCK = 128;

static const int STARTING_NUM_BLOCKS_IN_POOL = 64;

static std::vector<TrieNode> defaultEmptyVec = {};

VectorPool::VectorPool(int maxOutputSize) : maxOutputSize(maxOutputSize + ADDITIONAL_CAPACITY)
{}

std::vector<int> VectorPool::Acquire()
{
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        if (!vectorPool.empty()) {
            auto vec = std::move(vectorPool.front());
            vectorPool.pop_front();
            return vec;
        }
    }
    std::vector<int> newVec;
    newVec.reserve(maxOutputSize);
    return newVec;  // Return a new vector if none available
}

void VectorPool::Release(std::vector<int> &&vec)
{
    vec.clear();  // Clear vector contents before returning to the pool
    std::lock_guard<std::mutex> lock(poolMutex);
    vectorPool.push_back(vec);
}

TrieNodePool::TrieNodePool(TrieNodeBlockPool &blockPool) : blockPool(blockPool), nextNodeIdx(0), logger(GetLogger())
{
    currNodePool = blockPool.AcquireBlock();
}

TrieNode *TrieNodePool::AcquireNode()
{
    // nextNodeIdx points to the first unused node (it could also be out of range)
    if (nextNodeIdx < currNodePool.size()) {
        // The pool has already a node ready for use
        TrieNode *node = &currNodePool[nextNodeIdx];
        nextNodeIdx++;
        if (node->count != 0) {
            // In case the node was not already cleaned-up
            node->count = 0;
            if (node->children.bucket_count() > MAX_CHILDREN_OF_EMPTY_NODE_FIRST_BLOCK) {
                std::unordered_map<int32_t, TrieNode *>().swap(node->children);
            } else {
                node->children.clear();
            }
        }
        return node;
    }
    // The block is full, let's use another one
    nodePools.push_back(std::move(currNodePool));
    currNodePool = blockPool.AcquireBlock();  // we assume this was properly cleaned-up
    nextNodeIdx = 1;
    TrieNode *node = &currNodePool[0];
    if (node->count != 0) {
        // In case the node was not already cleaned-up
        node->count = 0;
        if (node->children.bucket_count() > MAX_CHILDREN_OF_EMPTY_NODE_FIRST_BLOCK) {
            std::unordered_map<int32_t, TrieNode *>().swap(node->children);
        } else {
            node->children.clear();
        }
    }
    return node;
}

/* When clearing up nodes, we just need to clean them up for the next use
and leave them in the pool. To avoid memory growing too much over time,
the nodes that had many children, and would still have large memory usage
after clearing the map, will have the children map deallocated too (but not
the root which is always big).*/
void TrieNodePool::ClearVector(std::vector<TrieNode> &block, size_t maxIndex, bool isFirstBlock)
{
    if (!block.empty()) {
        if (isFirstBlock) {
            // First element could is the root node -> clear map but never deallocate
            block[0].children.clear();
            block[0].count = 0;
            for (size_t i = 1; i < maxIndex; i++) {
                if (block[i].children.bucket_count() > MAX_CHILDREN_OF_EMPTY_NODE_FIRST_BLOCK) {
                    // Deallocate memory
                    std::unordered_map<int32_t, TrieNode *>().swap(block[i].children);
                } else {
                    block[i].children.clear();
                }
                block[i].count = 0;
            }
            return;
        }
        // Not the first block
        for (size_t i = 0; i < maxIndex; i++) {
            if (block[i].children.bucket_count() > MAX_CHILDREN_OF_EMPTY_NODE) {
                // Deallocate memory
                std::unordered_map<int32_t, TrieNode *>().swap(block[i].children);
            } else {
                block[i].children.clear();
            }
            block[i].count = 0;
        }
        return;
    }
    SPDLOG_LOGGER_WARN(logger, "Trying to clear an empty TrieNode block: there must be some implementation error!");
}

/* Returns all TrieNodeBlocks besides one (that will anyways be used again even if
the following sequence is shorter) to the TrieNodeBlockPool. */
void TrieNodePool::ClearPool(bool dontClearNodesIfNeeded)
{
    if (blockPool.poolEmpty.load(std::memory_order_relaxed) && dontClearNodesIfNeeded) {
        SPDLOG_LOGGER_DEBUG(logger, "Returning pool block without clearing!");
        // It's possible (although unlikely) that the cleaning thread cannot keep the pace of the put threads!
        // Don't waste time cleaning up data, let the put threads do it!
        if (!nodePools.empty()) {
            // The root block is the first of the deque: the current block can be returned
            // Here we use the (intelligent version that moves to the block pool)
            blockPool.ReleaseBlock(std::move(currNodePool));

            // Retrieve the first block (with root)
            currNodePool = std::move(nodePools.front());
            nodePools.pop_front();

            // Return to the block pool any additional blocks
            while (!nodePools.empty()) {
                std::vector<TrieNode> &currentBlock = nodePools.front();
                blockPool.ReleaseBlock(std::move(currentBlock));
                nodePools.pop_front();
            }
        }
        // Only clear the root
        currNodePool[0].children.clear();
        currNodePool[0].count = 0;
    } else {
        if (nodePools.empty()) {
            // There is only one block (with root)
            ClearVector(currNodePool, nextNodeIdx, true);
        } else {
            // The root block is the first of the deque: the current block can be returned
            ClearVector(currNodePool, nextNodeIdx, false);
            // For some weird reason this is faster
            blockPool.ReleaseBlockWithCopy(currNodePool);

            // Retrieve the first block and clear it
            currNodePool = std::move(nodePools.front());
            nodePools.pop_front();
            ClearVector(currNodePool, currNodePool.size(), true);

            // Return to the block pool any additional blocks
            while (!nodePools.empty()) {
                std::vector<TrieNode> &currentBlock = nodePools.front();
                ClearVector(currentBlock, currentBlock.size(), false);
                // For some weird reason this is faster
                blockPool.ReleaseBlockWithCopy(currentBlock);
                nodePools.pop_front();
            }
        }
    }
    // Will start again from the second node in currNodePool (the first remains root)
    nextNodeIdx = 1;
}

/* Returns all TrieNodeBlocks to the TrieNodeBlockPool. */
TrieNodePool::~TrieNodePool()
{
    SPDLOG_LOGGER_TRACE(logger, "Destroying the node pool!");
    ClearVector(currNodePool, nextNodeIdx, false);
    blockPool.ReleaseBlock(std::move(currNodePool));
    while (!nodePools.empty()) {
        std::vector<TrieNode> &currentBlock = nodePools.front();
        ClearVector(currentBlock, currentBlock.size(), false);
        blockPool.ReleaseBlock(std::move(currentBlock));
        nodePools.pop_front();
    }
}

TrieNodePool::TrieNodePool(TrieNodePool &&other) noexcept
    : blockPool(other.blockPool), currNodePool(std::move(other.currNodePool)), nextNodeIdx(other.nextNodeIdx),
      nodePools(std::move(other.nodePools)), logger(std::move(other.logger))
{}

TrieNodePool &TrieNodePool::operator=(TrieNodePool &&other) noexcept
{
    if (this != &other) {
        ClearPool(false);
        // the blockPool reference remains unchanged, was created at initialization
        currNodePool = std::move(other.currNodePool);
        nextNodeIdx = other.nextNodeIdx;
        nodePools = std::move(other.nodePools);
        logger = std::move(other.logger);
    }

    return *this;
}

TrieNodeBlockPool::TrieNodeBlockPool(size_t maxBlockSize) : maxBlockSize(maxBlockSize), logger(GetLogger())
{
    // Already fill the thread pool with some blocks. In case they are not enough more will be created.
    for (int i = 0; i < STARTING_NUM_BLOCKS_IN_POOL; i++) {
        blockPool.emplace_back(maxBlockSize);
    }
}

TrieNodeBlockPool::~TrieNodeBlockPool()
{
    SPDLOG_LOGGER_TRACE(logger, "Destroying TrieNodeBlockPool");
}

std::vector<TrieNode> TrieNodeBlockPool::AcquireBlock()
{
    // the pool-empty variable update is just to know if this thread pool is not
    // able to provide enough blocks (for backpressure handling): the checks are not meant
    // to be precise, they are not well synchronized, as we prioritize efficiency to
    // logical correctness: it is just a performance improvement, not application logic.
    bool wasPoolEmpty = poolEmpty.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        if (!blockPool.empty()) {
            auto block = std::move(blockPool.front());
            blockPool.pop_front();
            SPDLOG_LOGGER_DEBUG(logger, "Block aquired. TrieNodeBlockPool size: {}", blockPool.size());
            if (wasPoolEmpty) {  // avoid if possible the atomic-store call in the lock: it can be costly
                poolEmpty.store(false, std::memory_order_relaxed);
            }
            return block;
        }
    }
    SPDLOG_LOGGER_DEBUG(logger, "New node block created");
    if (!wasPoolEmpty) {
        poolEmpty.store(true, std::memory_order_relaxed);
    }
    return std::vector<TrieNode>(maxBlockSize);
}

void TrieNodeBlockPool::ReleaseBlock(std::vector<TrieNode> &&block)
{
    // The TrieNodes in the vectors inserted here should already be cleared (no children, count 0),
    // otherwise they will be cleaned when used
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        blockPool.push_back(std::move(block));
        SPDLOG_LOGGER_DEBUG(logger, "Block released. TrieNodeBlockPool size: {}", blockPool.size());
    }
}

void TrieNodeBlockPool::ReleaseBlockWithCopy(std::vector<TrieNode> block)
{
    // This method makes a copy when pushing to the deque, but for some weird reason this avoids
    // slowdowns on get methods happening in parallel!
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        blockPool.push_back(std::move(block));
        SPDLOG_LOGGER_DEBUG(logger, "Block released. TrieNodeBlockPool size: {}", blockPool.size());
    }
}

/* Dummy constructor to build the prompt cache in the beginning! NEVER USE A POOL CREATED THIS WAY!!!!! */
SequencePool::SequencePool()
    : branchLength(0), prefixLength(0), inputTokensToPutInSelfOutput(0),
      threadPool(nullptr), trieBlockPool(nullptr), vectorPool(nullptr)
{}

SequencePool::SequencePool(int branchLength, int prefixLength, int inputTokensToPutInSelfOutput, std::shared_ptr<ThreadPool> threadPool,
    std::shared_ptr<TrieNodeBlockPool> trieBlockPool, std::shared_ptr<VectorPool> vectorPool)
    : branchLength(branchLength), prefixLength(prefixLength), inputTokensToPutInSelfOutput(inputTokensToPutInSelfOutput),
      threadPool(threadPool), trieBlockPool(trieBlockPool), vectorPool(vectorPool)
{}

std::shared_ptr<Sequence> SequencePool::Acquire()
{
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        if (!sequencePool.empty()) {
            auto seq = sequencePool.front();
            sequencePool.pop_front();
            return seq;
        }
    }
    auto seq = std::make_shared<Sequence>(branchLength, prefixLength, inputTokensToPutInSelfOutput, threadPool, trieBlockPool, vectorPool);
    return seq;  // Return a new vector if none available
}

/* Does not require the sequence to be alreade cleared. */
void SequencePool::Release(std::shared_ptr<Sequence> seq, bool multipleWaiting)
{
    seq->Clear(multipleWaiting);  // Clear the sequence so that it is ready for the next use
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        sequencePool.push_back(seq);
    }
}

SequencePool &SequencePool::operator=(SequencePool &&other) noexcept
{
    if (this != &other) {
        branchLength = other.branchLength;
        prefixLength = other.prefixLength;
        inputTokensToPutInSelfOutput = other.inputTokensToPutInSelfOutput;
        threadPool = std::move(other.threadPool);
        trieBlockPool = std::move(other.trieBlockPool);
        vectorPool = std::move(other.vectorPool);

        // No need to move mutex as it's non-copyable, just leave it
        sequencePool = std::move(other.sequencePool);
    }
    return *this;
}