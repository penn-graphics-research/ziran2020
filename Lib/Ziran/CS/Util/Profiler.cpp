#include "Profiler.h"

#pragma once

#include <map>
#include <chrono>
#include <cmath>
#include <vector>
#include <sstream>
#include <iomanip>

namespace ZIRAN {

namespace PROFILER {

std::map<std::pair<std::string, int>, int> name_scope;

std::vector<std::pair<std::string, int>> scope_name(1, std::make_pair("Global", -1));
std::vector<std::chrono::duration<double>> scope_duration(1, std::chrono::duration<double>(0));
std::vector<std::chrono::duration<double>> global_duration(1, std::chrono::duration<double>(0));
std::vector<std::vector<int>> scope_edges(1, std::vector<int>());
std::vector<int> scope_stack(1, 0);

std::string duration2string(const std::chrono::duration<double>& elapsed_seconds)
{
    using namespace std::chrono;
    std::string str;
    auto dur = elapsed_seconds;

    int num_days = int(dur / hours(24));
    if (num_days) str += std::to_string(num_days) + "d ";
    dur -= num_days * hours(24);

    int num_hours = int(dur / hours(1));
    if (num_hours) str += std::to_string(num_hours) + "h ";
    dur -= num_hours * hours(1);

    int num_minutes = int(dur / minutes(1));
    if (num_minutes) str += std::to_string(num_minutes) + "m ";
    dur -= num_minutes * minutes(1);

    str += std::to_string(dur.count()) + "s";
    return str;
}

ScopedTimer::ScopedTimer(std::string name, bool enable)
    : enable(enable)
{
    if (!enable)
        return;
    start_time = std::chrono::steady_clock::now();
    auto name_parent = std::make_pair(name, scope_stack.back());
    if (name_scope.find(name_parent) != name_scope.end()) {
        id = name_scope[name_parent];
    }
    else {
        id = scope_name.size();
        name_scope[name_parent] = id;
        scope_name.push_back(name_parent);
        scope_duration.emplace_back(0);
        global_duration.emplace_back(0);
        scope_edges.emplace_back();
        scope_edges[scope_stack.back()].push_back(id);
    }
    scope_stack.push_back(id);
}

ScopedTimer::~ScopedTimer()
{
    if (!enable)
        return;
    end_time = std::chrono::steady_clock::now();
    scope_duration[id] += end_time - start_time;
    scope_stack.pop_back();
    // for (int i = 0; i < scope_stack.size(); ++i)
    //     printf("\t");
    // printf("%s\n", scope_name[id].first.c_str());
}

GlobalTimer::GlobalTimer()
{
    current_time = std::chrono::steady_clock::now();
}

GlobalTimer global_timer;

void traverseScopes(int id, int depth, std::stringstream& ss)
{
    if (depth < 0) {
        for (auto s : scope_edges[id])
            traverseScopes(s, depth + 1, ss);
        return;
    }
    auto printScope = [&]() {
        std::string scope_str = duration2string(scope_duration[id]);
        std::string global_str = duration2string(global_duration[id]);
        double scope_percent = scope_duration[id].count() * 100 / scope_duration[0].count();
        double global_percent = global_duration[id].count() * 100 / global_duration[0].count();
        ss << " " << scope_name[id].first.c_str() << " : " << scope_str.c_str() << " (" << scope_percent << "%)   \033[22;31m" << global_str.c_str() << " (" << global_percent << "%)\033[0m";
    };
    if (scope_edges[id].empty()) {
        if (!depth) return;
        for (int i = 0; i < depth; ++i) ss << "┃";
        printScope();
        ss << "\n";
    }
    else {
        for (int i = 0; i < depth; ++i) ss << "┃";
        ss << "┏";
        printScope();
        ss << "\n";
        for (auto s : scope_edges[id])
            traverseScopes(s, depth + 1, ss);

        // other scope
        std::chrono::duration<double> other_scope = scope_duration[id];
        std::chrono::duration<double> other_global = global_duration[id];
        for (auto s : scope_edges[id]) {
            other_scope -= scope_duration[s];
            other_global -= global_duration[s];
        }
        double other_scope_percent = other_scope.count() * 100 / scope_duration[0].count();
        double other_global_percent = other_global.count() * 100 / global_duration[0].count();
        if (other_global_percent >= 0.05) {
            for (int i = 0; i <= depth; ++i) ss << "┃";
            std::string scope_str = duration2string(other_scope);
            std::string global_str = duration2string(other_global);
            ss << " uncounted : " << scope_str.c_str() << " (" << other_scope_percent << "%)   \033[22;31m" << global_str.c_str() << " (" << other_global_percent << "%)\033[0m";
            ss << "\n";
        }

        for (int i = 0; i < depth; ++i) ss << "┃";
        ss << "┗\n";
    }
}

void flush(int depth)
{
    std::chrono::time_point<std::chrono::steady_clock> last_time = global_timer.current_time;
    std::chrono::time_point<std::chrono::steady_clock> current_time = std::chrono::steady_clock::now();
    global_timer.current_time = current_time;
    scope_duration[0] = current_time - last_time;
    for (int i = 0; i < (int)scope_duration.size(); ++i)
        global_duration[i] += scope_duration[i];
    std::stringstream ss;
    ss << std::setprecision(2) << std::fixed;
    std::string scope_str = duration2string(scope_duration[0]);
    std::string global_str = duration2string(global_duration[0]);
    ss << "global : " << scope_str.c_str() << " (100.00%)   \033[22;31m" << global_str.c_str() << " (100.00%)\033[0m\n";
    traverseScopes(0, depth, ss);
    ss << "\n";
    ZIRAN_INFO(ss.str().c_str());
    for (int i = 0; i < (int)scope_duration.size(); ++i)
        scope_duration[i] = std::chrono::duration<double>(0);
}
}

} // namespace ZIRAN::PROFILER
