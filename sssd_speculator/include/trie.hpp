/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#ifndef TRIE_HPP
#define TRIE_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <deque>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <utils.hpp>

namespace py = pybind11;

struct Slice {
    std::vector<int32_t>::const_iterator start;
    std::vector<int32_t>::const_iterator end;

    // Constructor
    Slice(std::vector<int32_t>::const_iterator start, std::vector<int32_t>::const_iterator end) : start(start), end(end)
    {}
};

struct TrieNode {
    std::unordered_map<int32_t, TrieNode *> children;
    int count;

    TrieNode();
    ~TrieNode();
};

/* MEMORY POOLING (implementation in memory_pooling.cpp)
This is necessary to avoid that, even if the memory is deallocated in a separate thread,
the memory deallocation causes delays on other threads doing time-sensitive tasks (retrieval
or insertion) */

/*
    Memory pool for vectors of the same size (for selfOutput).
*/
class VectorPool {
public:
    explicit VectorPool(int maxOutputSize);
    std::vector<int> Acquire();
    void Release(std::vector<int> &&vec);

private:
    std::mutex poolMutex;
    std::deque<std::vector<int>> vectorPool;
    size_t maxOutputSize;
};

/*
    Contains a collection of the vectors that will be used by TrieNodePools.
    Each sequence will have at least on of these.
    All the nodes in these vectors should be "cleared", which means, they must have no children.
    The clearing should be done before inserting in this pool.
*/
class TrieNodeBlockPool {
public:
    std::atomic<bool> poolEmpty;

    explicit TrieNodeBlockPool(size_t maxBlockSize);
    ~TrieNodeBlockPool();
    std::vector<TrieNode> AcquireBlock();
    void ReleaseBlock(std::vector<TrieNode> &&block);
    void ReleaseBlockWithCopy(std::vector<TrieNode> block);

    // Don't copy or assign
    TrieNodeBlockPool(const TrieNodeBlockPool &) = delete;
    TrieNodeBlockPool &operator=(const TrieNodeBlockPool &) = delete;

private:
    std::mutex poolMutex;
    std::deque<std::vector<TrieNode>> blockPool;
    size_t maxBlockSize;
    std::shared_ptr<spdlog::logger> logger;
};

/*
    This is not a normal thread pool: it is designed for efficiency taking advantage
    of our use case: it is not multi-threaded, and it assumes that we are always adding
    data to the tree, and releasing data all in once once a sequence is finished.
    It is a fixed-size vector, when you need a new node, if there is an unused node, it is
    used and the pointer to the first unused node gets increased. If all the vector is used
    (the size is fixed and specified at construction), then it aquires a new block from the
    TrieNodeBlockPool.
    This is to avoid reallocating or having too big pools in case of (rare) very long sequences.
    Releasing nodes simply consists of clearing their content.
*/
class TrieNodePool {
public:
    explicit TrieNodePool(TrieNodeBlockPool &blockPool);
    TrieNode *AcquireNode();
    void ClearVector(std::vector<TrieNode> &block, size_t maxIndex, bool isFirstBlock);
    /* Clears and returns all blocks but one to the TrieNodeBlockPool. If there are multiple sequences
    waiting for clearing and the blockPool ran out of blocks, does not clear the nodes (will be cleaned
    by who uses them). */
    void ClearPool(bool dontClearNodesIfNeeded);
    ~TrieNodePool();  // clears and returns all blocks to the TrieNodeBlockPool

    TrieNodePool(TrieNodePool &&other) noexcept;
    TrieNodePool &operator=(TrieNodePool &&other) noexcept;

private:
    TrieNodeBlockPool &blockPool;
    std::vector<TrieNode> currNodePool;
    // Index of the next node to use (of currNodePool)
    size_t nextNodeIdx;
    // Holds the pools that are still in use in the trie
    std::deque<std::vector<TrieNode>> nodePools;
    std::shared_ptr<spdlog::logger> logger;
};

// End memory pooling

// MAIN CLASSES

class Trie {
public:
    explicit Trie(int32_t stopToken, TrieNodeBlockPool &nodeBlocksPool);
    ~Trie();

    TrieNode *GetRoot();
    void Clear(bool dontClearNodesIfNeeded);
    void InsertSequence(const std::vector<int32_t> &sequence);
    void InsertSlice(const Slice &sequence);
    void InsertIter(const std::vector<int>::iterator &start, const std::vector<int>::iterator &end);
    void LogTrie() const
    {
        std::cout << "Trie Structure:\n";
        LogTrieNode(root, 0);
    }
    void LogTrieNode(TrieNode *node, int depth) const
    {
        if (!node)
            return;
        for (const auto &pair : node->children) {
            for (int i = 0; i < depth; ++i)
                std::cout << "  ";  // Indentation
            std::cout << "Token: " << pair.first << " (Count: " << pair.second->count << ")\n";
            LogTrieNode(pair.second, depth + 1);
        }
    }

    // Move constructor
    Trie(Trie &&other) noexcept;
    // Move assignment operator
    Trie &operator=(Trie &&other) noexcept;

private:
    TrieNode *root;
    int32_t stopToken;
    TrieNodePool nodePool;
};

struct FinalTrieNode {
    int32_t token;  // same value as the key of the parent that leads to this node. Needed for going backwards
    std::unordered_map<int32_t, FinalTrieNode *> children;
    std::unordered_map<int32_t, size_t> subtrees_depths;  // pairs of <token_id, max_depth> of all the children
    FinalTrieNode *parent;
    size_t depth;  // root has depth 0

    FinalTrieNode(int32_t token, FinalTrieNode *parent, size_t depth);
    ~FinalTrieNode();
};

class FinalTrie {
private:
    int startToken;
    FinalTrieNode *root;
    int totalNodes;
    int maxNodes;
    std::size_t maxChildrenPerNode;

    void dfs(const FinalTrieNode *node, std::vector<int> &path, std::vector<std::vector<int>> &results);

public:
    FinalTrie(int startToken, int maxNodes, std::size_t maxChildrenPerNode = std::numeric_limits<std::size_t>::max());
    ~FinalTrie();
    FinalTrieNode *Insert(FinalTrieNode *parent, int childValue);
    FinalTrieNode *GetRoot() const;
    int GetTotalNodes() const;
    std::vector<std::vector<int>> GetCartesianDrafts();
    void BuildCandidates(
        const FinalTrieNode *node, std::vector<int> &result, int parentIndex, std::vector<int> &parents);
    std::pair<std::vector<int>, std::vector<std::vector<bool>>> GetCandidatesAndAttnMask();
    std::pair<std::vector<int>, std::vector<int>> GetCandidatesAndAttnMaskRaw(std::shared_ptr<bool[]> &mask_buffer);
    std::tuple<
        std::vector<int>,  // candidates
        std::vector<int>,  // depths
        std::vector<int>,  // firstChildIdx
        std::vector<int>   // nextSiblingIdx
    > GetCandidatesMaskSglang(std::shared_ptr<bool[]>& mask_buffer);
};

struct ProbCandidateNode {
    int value;
    TrieNode *node;
    double prob;
    int depth;
    FinalTrieNode *precedingNode;
    double childDiscountFactor;

    ProbCandidateNode(int val, TrieNode *n, double p, int d, FinalTrieNode *pn, double childDiscountFactor);
    bool operator<(const ProbCandidateNode &other) const;
};

#endif  // TRIE_HPP