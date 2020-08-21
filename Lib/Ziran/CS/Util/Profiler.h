#pragma once

#include <Ziran/CS/Util/Logging.h>
#include <map>
#include <chrono>
#include <cmath>
#include <vector>

namespace ZIRAN {

namespace PROFILER {
/*
ATTRIBUTE_RESET = "\033[0m";
BLACK = "\033[22;30m";
RED = "\033[22;31m";
GREEN = "\033[22;32m";
BROWN = "\033[22;33m";
BLUE = "\033[22;34m";
MAGENTA = "\033[22;35m";
CYAN = "\033[22;36m";
GREY = "\033[22;37m";
DARKGREY = "\033[01;30m";
LIGHTRED = "\033[01;31m";
LIGHTGREEN = "\033[01;32m";
YELLOW = "\033[01;33m";
LIGHTBLUE = "\033[01;34m";
LIGHTMAGENTA = "\033[01;35m";
LIGHTCYAN = "\033[01;36m";
WHITE = "\033[01;37m";
*/

class ScopedTimer {
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::time_point<std::chrono::steady_clock> end_time;
    int id;
    bool enable;

public:
    ScopedTimer(std::string name, bool enable);
    ~ScopedTimer();
};

class GlobalTimer {
public:
    std::chrono::time_point<std::chrono::steady_clock> current_time;
    GlobalTimer();
};

extern GlobalTimer global_timer;

void flush(int depth);

} // namespace PROFILER

#define PROFILER_FLUSH(depth) PROFILER::flush(depth)

} // namespace ZIRAN
