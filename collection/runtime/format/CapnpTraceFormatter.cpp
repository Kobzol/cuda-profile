#include "CapnpTraceFormatter.h"

#ifdef CUPR_USE_ZLIB
    #include "zlib/zstr.hpp"
#endif

#ifdef CUPR_USE_CAPNP
    #include <capnp/message.h>
    #include <capnp/serialize-packed.h>
    #include <format/capnp/cupr.capnp.h>

    using namespace capnp;
    using namespace kj;

    class StdOutputStream: public OutputStream
    {
    public:
        explicit StdOutputStream(std::ostream& os): os(os)
        {

        }

        void write(const void* buffer, size_t size) final
        {
            this->os.write(static_cast<const char*>(buffer), size);
            assert(this->os);
        }

    private:
        std::ostream& os;
    };
#endif

#define WRAP_STR(s) (s == nullptr ? "" : s)

void cupr::CapnpTraceFormatter::formatTrace(std::ostream& os, const std::string& kernel, DeviceDimensions dimensions,
                                            const std::vector<cupr::Warp>& warps,
                                            const std::vector<cupr::AllocRecord>& allocations, double start, double end,
                                            bool prettify, bool compress)
{
#ifdef CUPR_USE_CAPNP
    MallocMessageBuilder message;
    capcupr::Trace::Builder traceBuilder = message.initRoot<capcupr::Trace>();
    traceBuilder.setType("trace");
    traceBuilder.setBankSize(dimensions.bankSize);
    traceBuilder.setWarpSize(dimensions.warpSize);

    auto blockDim = traceBuilder.initBlockDim();
    blockDim.setX(dimensions.block.x);
    blockDim.setY(dimensions.block.y);
    blockDim.setZ(dimensions.block.z);

    auto gridDim = traceBuilder.initGridDim();
    gridDim.setX(dimensions.grid.x);
    gridDim.setY(dimensions.grid.y);
    gridDim.setZ(dimensions.grid.z);

    traceBuilder.setKernel(kernel);
    traceBuilder.setStart(start);
    traceBuilder.setEnd(end);

    List<capcupr::AllocRecord>::Builder allocationsBuilder = traceBuilder.initAllocations(
            static_cast<unsigned int>(allocations.size()));

    for (int i = 0; i < static_cast<int>(allocations.size()); i++)
    {
        auto& alloc = allocations[i];
        capcupr::AllocRecord::Builder allocBuilder = allocationsBuilder[i];
        allocBuilder.setAddress(this->hexPointer(alloc.address));
        allocBuilder.setSpace(static_cast<uint8_t>(alloc.addressSpace));
        allocBuilder.setSize(alloc.size);
        allocBuilder.setElementSize(static_cast<uint32_t>(alloc.elementSize));
        allocBuilder.setActive(alloc.active);
        allocBuilder.setLocation(WRAP_STR(alloc.location));
        allocBuilder.setTypeIndex(alloc.typeIndex);
        allocBuilder.setTypeString(WRAP_STR(alloc.type));
        allocBuilder.setNameIndex(alloc.nameIndex);
        allocBuilder.setNameString(WRAP_STR(alloc.name));
    }

    List<capcupr::Warp>::Builder warpsBuilder = traceBuilder.initWarps(static_cast<unsigned int>(warps.size()));

    for (int w = 0; w < warps.size(); w++)
    {
        capcupr::Warp::Builder warpBuilder = warpsBuilder[w];
        auto& warp = warps[w];

        List<capcupr::MemoryAccess>::Builder accessList = warpBuilder.initAccesses(
                static_cast<unsigned int>(warp.accesses.size()));
        capcupr::Dim3::Builder blockId = warpBuilder.initBlockIdx();
        blockId.setX(warp.blockIndex.x);
        blockId.setY(warp.blockIndex.y);
        blockId.setZ(warp.blockIndex.z);

        warpBuilder.setDebugId(warp.debugId);
        warpBuilder.setKind(warp.accessType);
        warpBuilder.setSize(warp.size);
        warpBuilder.setSpace(warp.space);
        warpBuilder.setTypeIndex(warp.typeId);
        warpBuilder.setWarpId(warp.id);
        warpBuilder.setTimestamp(std::to_string(warp.timestamp));

        for (int i = 0; i < warp.accesses.size(); i++)
        {
            auto& access = warp.accesses[i];
            capcupr::MemoryAccess::Builder accessBuilder = accessList[i];
            accessBuilder.setValue(this->hexPointer(reinterpret_cast<const void*>(access.value)));
            accessBuilder.setAddress(this->hexPointer(reinterpret_cast<const void*>(access.address)));

            capcupr::Dim3::Builder threadIndex = accessBuilder.initThreadIdx();
            threadIndex.setX(access.threadIndex.x);
            threadIndex.setY(access.threadIndex.y);
            threadIndex.setZ(access.threadIndex.z);
        }
    }

    auto out = this->createStream(os, compress);
    StdOutputStream stream(*out);
    writePackedMessage(stream, message);
#endif
}

std::string cupr::CapnpTraceFormatter::getSuffix()
{
    return "capnp";
}

std::unique_ptr<std::ostream> cupr::CapnpTraceFormatter::createStream(std::ostream& os, bool compress)
{
#ifdef CUPR_USE_ZLIB
    if (compress)
    {
        return std::make_unique<zstr::ostream>(os);
    }
#endif
    return std::make_unique<std::ostream>(os.rdbuf());
}
