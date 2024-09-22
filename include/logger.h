#include <iostream>
#include <fstream>
#include <string>

enum class LogLevel { INFO, WARNING, ERROR };

class Logger {
public:
    Logger(const std::string& filename) : ofs(filename, std::ios::out | std::ios::app) {}

    void log(LogLevel level, const std::string& message) {
        if (level >= log_level) {
            ofs << to_string(level) << ": " << message << std::endl;
        }
    }

    void set_log_level(LogLevel level) { log_level = level; }

private:
    std::string to_string(LogLevel level) {
        switch (level) {
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    LogLevel log_level = LogLevel::INFO;
    std::ofstream ofs;
};
