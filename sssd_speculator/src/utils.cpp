/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include <utils.hpp>

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