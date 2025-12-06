// 公用函数
#ifndef __UTILS_H__
#define __UTILS_H__

#include "spdlog/spdlog.h"
#include "yaml-cpp/yaml.h"

#include <string>
#include <unordered_map>
#include <vector>
#include "random"

using namespace std;

// 从yaml中获取模型的类映射
unordered_map<int, string> get_class_name(const string &path) {
    unordered_map<int, string> class_name_map;
    YAML::Node config = YAML::LoadFile(path);
    if (config["names"] && config["names"].IsMap()) {
        const YAML::Node &names_node = config["names"];
        for (YAML::const_iterator it = names_node.begin(); it != names_node.end(); ++it) {
            int class_id = it->first.as<int>();           // 转换为int
            string class_name = it->second.as<string>();  // 转换为string
            class_name_map[class_id] = class_name;
        }
    } else {
        spdlog::critical("yaml don't have class name map");
    }
    return class_name_map;
}

// 根据类别个数随即生成颜色
unordered_map<int,array<int,3>>  gen_color_map(const int nums){
    unordered_map<int,array<int,3>> color_map;
    // 初始化随机数生成器
    static random_device rd; // 获取随机种子
    static mt19937 gen(rd()); // 梅森扭结器，高性能随机数引擎
    uniform_int_distribution<> dis(0,255); // 定义0~255的均匀分布
    for (int i=0;i<nums;++i){
        int r = dis(gen);
        int g = dis(gen);
        int b = dis(gen);
        color_map[i] = {r,g,b};
    }
    return color_map;
}

#endif  // __UTILS_H__