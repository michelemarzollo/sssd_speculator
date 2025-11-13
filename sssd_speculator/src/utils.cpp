/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include <utils.hpp>
#include <pthread.h>

#ifdef HAVE_LIBNUMA
  #include <numa.h>
  #include <numaif.h>
#endif

// Singleton logger
std::shared_ptr<spdlog::logger> GetLogger()
{
    static std::shared_ptr<spdlog::logger> logger = nullptr;

    static std::once_flag init_flag;

    std::call_once(init_flag, []() {
        // Create the logger only once
        logger = spdlog::stdout_color_mt("sssd_speculator");
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] [%s:%#] %v");

        // Set the logging level based on SPDLOG_ACTIVE_LEVEL
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
        logger->set_level(spdlog::level::trace);
#elif SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
        logger->set_level(spdlog::level::debug);
#endif
    });
    return logger;
}

std::size_t GetPinnedCpuCount()
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

void PinThreadToCore(int coreId)
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

// NUMA restrictions

int GetCurrentNUMANode(std::shared_ptr<spdlog::logger> logger)
{
#ifdef HAVE_LIBNUMA
    if (numa_available() == -1) return -1;
    int cpu = sched_getcpu();
    if (cpu < 0) { SPDLOG_LOGGER_ERROR(logger, "sched_getcpu() failed."); return -1; }
    int node = numa_node_of_cpu(cpu);
    if (node < 0) { SPDLOG_LOGGER_ERROR(logger, "numa_node_of_cpu({}) failed.", cpu); return -1; }
    return node;
#else
    (void)logger;
    return -1;
#endif
}

bool BuildCpusetForNode(int node, cpu_set_t &out, std::shared_ptr<spdlog::logger> logger)
{
    CPU_ZERO(&out);
#ifdef HAVE_LIBNUMA
    if (numa_available() == -1 || node < 0) return false;

    bitmask *nodeMask = numa_allocate_cpumask();
    if (!nodeMask) { SPDLOG_LOGGER_ERROR(logger, "numa_allocate_cpumask() failed."); return false; }
    if (numa_node_to_cpus(node, nodeMask) != 0) {
        SPDLOG_LOGGER_ERROR(logger, "numa_node_to_cpus(node={}) failed.", node);
        numa_free_cpumask(nodeMask);
        return false;
    }

    cpu_set_t curAffinity; CPU_ZERO(&curAffinity);
    if (sched_getaffinity(0, sizeof(curAffinity), &curAffinity) != 0) {
        SPDLOG_LOGGER_ERROR(logger, "sched_getaffinity() failed.");
        numa_free_cpumask(nodeMask);
        return false;
    }

    for (int cpu = 0; cpu < (int)nodeMask->size; ++cpu) {
        if (numa_bitmask_isbitset(nodeMask, cpu) && CPU_ISSET(cpu, &curAffinity)) {
            CPU_SET(cpu, &out);
        }
    }
    numa_free_cpumask(nodeMask);
    return CPU_COUNT(&out) > 0;
#else
    (void)node; (void)logger;
    return false;
#endif
}

bool PinThreadToLocalNUMANode(std::shared_ptr<spdlog::logger> logger)
{
#ifdef HAVE_LIBNUMA
    int node = GetCurrentNUMANode(logger);
    if (node < 0) return false;
    cpu_set_t set;
    if (!BuildCpusetForNode(node, set, logger)) return false;
    pthread_t th = pthread_self();
    int ret = pthread_setaffinity_np(th, sizeof(set), &set);
    if (ret != 0) { SPDLOG_LOGGER_ERROR(logger, "pthread_setaffinity_np() failed: {}", ret); return false; }
    return true;
#else
    (void)logger;
    return false;
#endif
}

void PreferLocalNodeMemory(std::shared_ptr<spdlog::logger> logger)
{
#ifdef HAVE_LIBNUMA
    // Avoid printing issue of being unable to set memory
    if (numa_available() == -1) return;

    int node = GetCurrentNUMANode(logger);
    if (node < 0) return;

    int maxnode = numa_max_node() + 1;
    if (maxnode <= 0) return;

    // Build a nodemask with 1 bit only (current node)
    const int bits_per_ulong = sizeof(unsigned long) * 8;
    std::vector<unsigned long> mask((maxnode + bits_per_ulong - 1) / bits_per_ulong, 0UL);
    mask[ node / bits_per_ulong ] |= (1UL << (node % bits_per_ulong));

    // MPOL_PREFERRED = "soft" preference (no hard bind). Not printing to stderr.
    int rc = set_mempolicy(MPOL_PREFERRED, mask.data(), maxnode);
    if (rc != 0) {
        if (errno == EPERM || errno == EACCES || errno == ENOSYS) {
            SPDLOG_LOGGER_DEBUG(logger, "mempolicy preferred blocked ({}). Continuing without it.", strerror(errno));
        } else {
            SPDLOG_LOGGER_DEBUG(logger, "set_mempolicy(MPOL_PREFERRED) failed: {}", strerror(errno));
        }
    }
#else
    (void)logger;
#endif
}

bool ConstrainProcessToLocalNode(std::shared_ptr<spdlog::logger> logger)
{
#ifdef HAVE_LIBNUMA
    if (numa_available() == -1) return false;

    int node = GetCurrentNUMANode(logger);
    if (node < 0) return false;

    cpu_set_t set;
    if (!BuildCpusetForNode(node, set, logger)) return false;

    // 1) CPU: limit current thread; new threads should also pin themselves
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        SPDLOG_LOGGER_ERROR(logger, "sched_setaffinity() failed: {}", strerror(errno));
        return false;
    }

    // 2) Memory: soft bind to this node for the calling thread (avoid error logs);
    int maxnode = numa_max_node() + 1;
    const int bits_per_ulong = sizeof(unsigned long) * 8;
    std::vector<unsigned long> mask((maxnode + bits_per_ulong - 1) / bits_per_ulong, 0UL);
    mask[ node / bits_per_ulong ] |= (1UL << (node % bits_per_ulong));

    int rc = set_mempolicy(MPOL_PREFERRED, mask.data(), maxnode);
    if (rc != 0) {
        if (errno == EPERM || errno == EACCES || errno == ENOSYS) {
            SPDLOG_LOGGER_DEBUG(logger, "mempolicy preferred blocked ({}). Proceeding with CPU affinity only.", strerror(errno));
        } else {
            SPDLOG_LOGGER_DEBUG(logger, "set_mempolicy(MPOL_PREFERRED) failed: {}", strerror(errno));
        }
    }
    return true;
#else
    (void)logger;
    return false;
#endif
}

bool ApplyThreadAffinity(const cpu_set_t& set, std::shared_ptr<spdlog::logger> logger)
{
    pthread_t th = pthread_self();
    int ret = pthread_setaffinity_np(th, sizeof(set), &set);
    if (ret != 0) {
        SPDLOG_LOGGER_ERROR(logger, "pthread_setaffinity_np() failed: {}", ret);
        return false;
    }
    SPDLOG_LOGGER_DEBUG(logger, "Thread affinity applied ({} CPUs).", CPU_COUNT(&set));
    return true;
}

bool BuildEffectiveCpuset(
    AffinityScope scope,
    cpu_set_t& out,
    int core_id /* used only for kSpecificCore */,
    std::shared_ptr<spdlog::logger> logger)
{
    CPU_ZERO(&out);

    // Start from current affinity (respect cgroups/taskset/docker constraints)
    cpu_set_t cur;
    CPU_ZERO(&cur);
    if (sched_getaffinity(0, sizeof(cur), &cur) != 0) {
        SPDLOG_LOGGER_ERROR(logger, "sched_getaffinity() failed.");
        return false;
    }

    if (scope == AffinityScope::kAny) {
        out = cur;
        return CPU_COUNT(&out) > 0;
    }

#ifdef HAVE_LIBNUMA
    const bool numa_ok = (numa_available() != -1);
#else
    const bool numa_ok = false;
#endif

    if (scope == AffinityScope::kLocalNode) {
        if (!numa_ok) {
            // Non-NUMA system: nothing special to do
            out = cur;
            return CPU_COUNT(&out) > 0;
        }
#ifdef HAVE_LIBNUMA
        int cpu = sched_getcpu();
        if (cpu < 0) {
            SPDLOG_LOGGER_ERROR(logger, "sched_getcpu() failed.");
            return false;
        }
        int node = numa_node_of_cpu(cpu);
        if (node < 0) {
            SPDLOG_LOGGER_ERROR(logger, "numa_node_of_cpu({}) failed.", cpu);
            return false;
        }

        bitmask* mask = numa_allocate_cpumask();
        if (!mask) {
            SPDLOG_LOGGER_ERROR(logger, "numa_allocate_cpumask() failed.");
            return false;
        }
        if (numa_node_to_cpus(node, mask) != 0) {
            SPDLOG_LOGGER_ERROR(logger, "numa_node_to_cpus(node={}) failed.", node);
            numa_free_cpumask(mask);
            return false;
        }

        // Intersect node cpus with current affinity
        for (int i = 0; i < (int)mask->size; ++i) {
            if (numa_bitmask_isbitset(mask, i) && CPU_ISSET(i, &cur)) {
                CPU_SET(i, &out);
            }
        }
        numa_free_cpumask(mask);

        if (CPU_COUNT(&out) == 0) {
            SPDLOG_LOGGER_WARN(logger, "No CPUs left after intersecting with NUMA node; using current affinity.");
            out = cur; // graceful fallback
        }
        return CPU_COUNT(&out) > 0;
#endif
    }

    // kSpecificCore
    {
        // Validate the core is in current affinity
        if (!CPU_ISSET(core_id, &cur)) {
            SPDLOG_LOGGER_ERROR(logger, "Core {} not in current affinity.", core_id);
            return false;
        }

#ifdef HAVE_LIBNUMA
        if (numa_ok) {
            int cpu_here = sched_getcpu();
            if (cpu_here < 0) {
                SPDLOG_LOGGER_ERROR(logger, "sched_getcpu() failed.");
                return false;
            }
            int node_here = numa_node_of_cpu(cpu_here);
            int node_core = numa_node_of_cpu(core_id);
            if (node_here < 0 || node_core < 0) {
                SPDLOG_LOGGER_ERROR(logger, "numa_node_of_cpu() failed.");
                return false;
            }
            if (node_here != node_core) {
                SPDLOG_LOGGER_ERROR(logger, "Core {} is not in local NUMA node {}.", core_id, node_here);
                return false;
            }
        }
#endif
        CPU_ZERO(&out);
        CPU_SET(core_id, &out);
        return true;
    }
}

std::vector<int> GetEffectiveCores(AffinityScope scope, std::shared_ptr<spdlog::logger> logger, int core_id)
{
    std::vector<int> outv;
    cpu_set_t set;
    if (!BuildEffectiveCpuset(scope, set, core_id, logger)) return outv;

    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &set)) outv.push_back(i);
    }
    return outv;
}
