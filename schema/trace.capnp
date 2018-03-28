@0xdb50473b24db4f29;

using Cxx = import "c++.capnp";
$Cxx.namespace("capcupr");

struct Dim3 {
    x @0 :Int32;
    y @1 :Int32;
    z @2 :Int32;
}

struct MemoryAccess {
    threadIdx @0 :Dim3;
    address @1 :Text;
    value @2 :Text;
}

struct Warp {
    accesses @0 :List(MemoryAccess);
    blockIdx @1 :Dim3;
    warpId @2 :Int32;
    debugId @3 :Int32;
    size @4 :UInt8;
    kind @5 :UInt8;
    space @6 :UInt8;
    typeIndex @7 :Int32;
    timestamp @8 :Text;
}

struct AllocRecord {
    address @0: Text;
    size @1: UInt64;
    elementSize @2: UInt32;
    space @3: UInt8;
    typeIndex @4: Int32;
    typeString @5: Text;
    nameIndex @6: Int32;
    nameString @7: Text;
    location @8: Text;
    active @9: Bool;
}

struct Trace {
    warps @0 :List(Warp);
    allocations @1: List(AllocRecord);
    kernel @2: Text;
    start @3: Float64;
    end @4: Float64;
    type @5: Text;
    gridDim @6: Dim3;
    blockDim @7: Dim3;
    warpSize @8: UInt32;
    bankSize @9: UInt32;
}
