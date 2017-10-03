#pragma once

#include <iostream>

namespace llvm {
    class DIVariable;
}

struct DebugInfo
{
public:
    DebugInfo() = default;
    explicit DebugInfo(std::string name, std::string filename = "", int line = -1);
    explicit DebugInfo(const llvm::DIVariable* debugVariable);

    bool hasName() const;
    std::string getName() const;
    std::string getFilename() const;
    int getLine() const;
    bool isValid() const;

    explicit operator bool() const;

    void print(std::ostream& os = std::cerr) const;

private:
    std::string name;
    std::string filename;
    int line = -1;
    bool valid = false;
};

std::ostream& operator<<(std::ostream& o, const DebugInfo& info);
