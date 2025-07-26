/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include <trie.hpp>
#include <queue>
#include <algorithm>


TrieNode::TrieNode() : count(0)
{}

TrieNode::~TrieNode()
{
    // TrieNodes in Tries, differently from FinalTrie, are taken from vectors (pools) and not allocated with new,
    // therefore they don't need to be deleted. Don't use new to create TrieNodes!
}

// Implementations of Trie
Trie::Trie(int32_t stopToken, TrieNodeBlockPool &nodeBlocksPool) : stopToken(stopToken), nodePool(nodeBlocksPool)
{
    root = nodePool.AcquireNode();
}

Trie::~Trie()
{
    // Breaks all connections between nodes and returns all nodePools to the blockPool, by calling the TrieNodePool
    // destructor (nodes don't get deallocated). All the rest gets destroyed
}

void Trie::Clear(bool dontClearNodesIfNeeded)
{
    // Differently from the destructor, this method keeps the first block of TrieNodes for the next usage in the
    // nodePool
    nodePool.ClearPool(dontClearNodesIfNeeded);
}

TrieNode *Trie::GetRoot()
{
    return root;
}

void Trie::InsertSequence(const std::vector<int32_t> &sequence)
{
    TrieNode *current_node = root;
    ++current_node->count;
    for (int32_t num : sequence) {
        if (num == stopToken) {
            return;
        }
        if (current_node->children.find(num) == current_node->children.end()) {
            current_node->children[num] = nodePool.AcquireNode();
        }
        current_node = current_node->children[num];
        ++current_node->count;
    }
}

void Trie::InsertSlice(const Slice &slice)
{
    TrieNode *current_node = root;
    ++current_node->count;
    for (auto it = slice.start; it < slice.end; ++it) {
        int32_t num = *it;
        if (num == stopToken) {
            return;
        }
        if (current_node->children.find(num) == current_node->children.end()) {
            current_node->children[num] = nodePool.AcquireNode();
        }
        current_node = current_node->children[num];
        ++current_node->count;
    }
}

// used by the lookahead cache
void Trie::InsertIter(const std::vector<int>::iterator &start, const std::vector<int>::iterator &end)
{
    TrieNode *current_node = root;
    ++current_node->count;
    for (auto it = start; it < end; ++it) {
        int32_t num = *it;
        if (current_node->children.find(num) == current_node->children.end()) {
            current_node->children[num] = nodePool.AcquireNode();
        }
        current_node = current_node->children[num];
        ++current_node->count;
    }
}

Trie::Trie(Trie &&other) noexcept : stopToken(other.stopToken), nodePool(std::move(other.nodePool)), root(other.root)
{
    other.root = nullptr;
}

Trie &Trie::operator=(Trie &&other) noexcept
{
    if (this != &other) {
        Clear(false);
        stopToken = other.stopToken;
        nodePool = std::move(other.nodePool);
        root = other.root;
        other.root = nullptr;
    }
    return *this;
}

// Implementation of FinalTrie

FinalTrieNode::FinalTrieNode(int32_t token, FinalTrieNode *parent, size_t depth)
    : token(token), parent(parent), depth(depth)
{}

FinalTrieNode::~FinalTrieNode()
{
    for (auto &child : children) {
        delete child.second;
    }
}

FinalTrie::FinalTrie(int startToken, int maxNodes, std::size_t maxChildrenPerNode)
    : startToken(startToken), root(new FinalTrieNode(startToken, nullptr, 0)), totalNodes(1),
    maxNodes(maxNodes), maxChildrenPerNode(maxChildrenPerNode)
{}

FinalTrie::~FinalTrie()
{
    delete root;
}

FinalTrieNode *FinalTrie::Insert(FinalTrieNode *parent, int childValue)
{
    if (parent->children.find(childValue) == parent->children.end()) {
        if (parent->children.size() >= maxChildrenPerNode) {
            return nullptr;
        }
        totalNodes++;
        size_t current_depth = parent->depth + 1;
        parent->children[childValue] = new FinalTrieNode(childValue, parent, current_depth);
        // Go backwards to update all the max subtrees depths for each node
        // Direct parent is easy
        parent->subtrees_depths[childValue] = 1;
        // Go to grandparent
        FinalTrieNode *next_parent = parent->parent;
        int32_t prev_child = parent->token;
        while (next_parent != nullptr) {
            size_t new_subtree_depth = current_depth - next_parent->depth;
            if (next_parent->subtrees_depths[prev_child] < new_subtree_depth) {
                next_parent->subtrees_depths[prev_child] = new_subtree_depth;
            }
            prev_child = next_parent->token;
            next_parent = next_parent->parent;
        }
    }
    // else the node was already present in the trie
    return parent->children[childValue];
}

void FinalTrie::dfs(const FinalTrieNode *node, std::vector<int> &path, std::vector<std::vector<int>> &results)
{
    if (node->children.empty()) {
        results.push_back(path);
        return;
    }
    for (auto &child : node->children) {
        path.push_back(child.first);
        dfs(child.second, path, results);
        path.pop_back();
    }
}

void FinalTrie::BuildCandidates(
    const FinalTrieNode *node, std::vector<int> &result, int parentIndex, std::vector<int> &parents)
{
    // This a depth-first search were we want to navigate first the longest branches!!
    std::vector<std::pair<int32_t, size_t>> branches(node->subtrees_depths.begin(), node->subtrees_depths.end());
    // sort the children based on how long the corresponding subtree is
    std::sort(branches.begin(), branches.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;  // Compare based on the values
    });
    for (const auto &pair : branches) {
        int32_t token_id = pair.first;
        result.push_back(token_id);
        parents.push_back(parentIndex);
        BuildCandidates(node->children.at(token_id), result, parents.size() - 1, parents);
    }
}

std::pair<std::vector<int>, std::vector<std::vector<bool>>> FinalTrie::GetCandidatesAndAttnMask()
{
    // Returns the mask as a vector of bool. This also includes the last token of the sequence in the mask
    // (the first column will be of trues).
    int n = GetTotalNodes();
    std::vector<int> candidates;
    candidates.reserve(n);
    candidates.push_back(startToken);
    std::vector<int> parents;
    parents.reserve(n);
    parents.push_back(-1);

    BuildCandidates(root, candidates, 0, parents);
    std::vector<std::vector<bool>> mask(n, std::vector<bool>(n, false));
    for (int i = 0; i < n; ++i) {
        mask[i][i] = true;
        int parentIndex = parents[i];
        for (int j = 0; j <= parentIndex; j++) {
            if (mask[parentIndex][j] == true) {
                mask[i][j] = true;
            }
        }
    }

    return std::make_pair(candidates, mask);
}

std::pair<std::vector<int>, std::vector<int>> FinalTrie::GetCandidatesAndAttnMaskRaw(
    std::shared_ptr<bool[]> &mask_buffer)
{
    // Returns the mask as a vector of bools. This does not include the last token of the sequence in the mask
    // (the first column will be of contain falses). The candidates, instead, include also the last token in the
    // sequence.
    int n = GetTotalNodes();
    std::vector<int> candidates;
    candidates.reserve(n);
    candidates.push_back(startToken);
    std::vector<int> parents;
    parents.reserve(n);
    parents.push_back(-1);
    std::vector<int> depths(n, 1);
    depths[0] = 0;  // root has depth 0

    BuildCandidates(root, candidates, 0, parents);

    // the mask will not include the root node
    // Remove one element and shift all parent indices
    n = n - 1;
    parents.erase(parents.begin());
    for (int &parentIndex : parents) {
        parentIndex--;  // those that had the root as parent, now will have -1 and will be excluded in the inner for
                        // loop
    }

    mask_buffer = std::shared_ptr<bool[]>(new bool[n * n], std::default_delete<bool[]>());
    std::fill(mask_buffer.get(), mask_buffer.get() + n * n, false);
    for (int i = 0; i < n; ++i) {
        mask_buffer[i * n + i] = true;
        int parentIndex = parents[i];
        for (int j = 0; j <= parentIndex; j++) {
            if (mask_buffer[parentIndex * n + j]) {
                mask_buffer[i * n + j] = true;
                depths[i + 1]++;
            }
        }
    }

    // Return candidates, raw pointer to mask, and dimensions
    return std::make_pair(candidates, depths);
}

// For SGLang

std::tuple<std::vector<int>, std::vector<int>,
    std::vector<int>, std::vector<int>> FinalTrie::GetCandidatesMaskSglang(
    std::shared_ptr<bool[]>& mask_buffer)
{
    /* SGLang requires a mask built with BFS. */
    
    const int nTot = GetTotalNodes();

    std::vector<int> candidates;        candidates.reserve(nTot);
    std::vector<int> depths;            depths.reserve(nTot);
    std::vector<int> firstChildIdx;     firstChildIdx.reserve(nTot);
    std::vector<int> nextSiblingIdx;    nextSiblingIdx.reserve(nTot);
    std::vector<int> parents;           parents.reserve(nTot);

    // root (index 0 in every vector)
    candidates.push_back(startToken);
    depths.push_back(0);
    firstChildIdx.push_back(-1);
    nextSiblingIdx.push_back(-1);
    parents.push_back(-1);

    std::queue<std::pair<const FinalTrieNode*, int>> q;
    q.emplace(root, 0);

    while (!q.empty())
    {
        const std::size_t levelSize = q.size();  // nodes currently in queue
        int prevOnLevel = -1;   // for nextSibling links

        for (std::size_t i = 0; i < levelSize; ++i)
        {
            auto front = q.front();
            q.pop();
            const FinalTrieNode* node = front.first;
            int nodeIdx = front.second;

            // sibling chain for the current level
            if (prevOnLevel != -1)
                nextSiblingIdx[prevOnLevel] = nodeIdx;
            prevOnLevel = nodeIdx;

            // Gather children, longest sub-tree first (usually longer branches mean higher prob, and faster acceptance)
            std::vector<std::pair<int32_t, std::size_t>> branches(
                    node->subtrees_depths.begin(), node->subtrees_depths.end());
            std::sort(branches.begin(), branches.end(),
                      [](const auto& a, const auto& b){ return a.second > b.second; });

            int firstChildForNode = -1;

            for (const auto& br : branches)
            {
                int32_t tok = br.first;
                const FinalTrieNode*  child = node->children.at(tok);

                const int childIdx = static_cast<int>(candidates.size());

                if (firstChildForNode == -1)
                    firstChildForNode = childIdx;   // remember first child

                // Push child into the BFS order
                candidates.push_back(tok);
                depths.push_back(depths[nodeIdx] + 1);
                firstChildIdx.push_back(-1);            // to be filled later
                nextSiblingIdx.push_back(-1);
                parents.push_back(nodeIdx);

                q.emplace(child, childIdx);
            }

            if (firstChildForNode != -1)
                firstChildIdx[nodeIdx] = firstChildForNode;
        }
    }

    const int N = nTot;
    mask_buffer.reset(new bool[N * N]{});

    for (int i = 0; i < N; ++i)
    {
        mask_buffer[i * N + i] = true;

        for (int p = parents[i]; p >= 0; p = parents[p])
            mask_buffer[i * N + p] = true;
    }

    return {
        std::move(candidates),
        std::move(depths),
        std::move(firstChildIdx),
        std::move(nextSiblingIdx)
        };
}

std::vector<std::vector<int>> FinalTrie::GetCartesianDrafts()
{
    std::vector<std::vector<int>> drafts;
    std::vector<int> current_path;
    dfs(root, current_path, drafts);
    return drafts;
}

int FinalTrie::GetTotalNodes() const
{
    return totalNodes;
}

FinalTrieNode *FinalTrie::GetRoot() const
{
    return root;
}

ProbCandidateNode::ProbCandidateNode(
    int val, TrieNode *n, double p, int d, FinalTrieNode *pn, double childDiscountFactor)
    : value(val), node(n), prob(p), depth(d), precedingNode(pn), childDiscountFactor(childDiscountFactor)
{}

bool ProbCandidateNode::operator<(const ProbCandidateNode &other) const
{
    return prob < other.prob || (prob == other.prob && depth < other.depth);
}