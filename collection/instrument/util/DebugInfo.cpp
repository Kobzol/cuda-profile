#include "DebugInfo.h"

#include <llvm/IR/DebugInfoMetadata.h>

DebugInfo::DebugInfo(std::string name, std::string filename, int line)
        : name(std::move(name)), filename(std::move(filename)), line(line), valid(!this->name.empty())
{

}
DebugInfo::DebugInfo(const llvm::DIVariable* debugVariable)
        : name(debugVariable->getName()), filename(debugVariable->getFilename()), line(debugVariable->getLine()), valid(true)
{

}

bool DebugInfo::hasName() const
{
    return !this->getName().empty();
}
std::string DebugInfo::getName() const
{
    return this->name;
}
std::string DebugInfo::getFilename() const
{
    return this->filename;
}
int DebugInfo::getLine() const
{
    return this->line;
}

bool DebugInfo::isValid() const
{
    return this->valid;
}
DebugInfo::operator bool() const
{
    return this->isValid();
}

void DebugInfo::print(std::ostream& os) const
{
    if (this->valid)
    {
        os << this->name << " at " << this->filename << ":" << this->line << std::endl;
    }
    else os << "Invalid location" << std::endl;
}

std::ostream& operator<<(std::ostream& o, const DebugInfo& info)
{
    info.print(o);
    return o;
}
