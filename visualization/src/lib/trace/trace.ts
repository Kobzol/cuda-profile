export interface Trace
{
    type: string;
    kernel: string;
    start: number;
    end: number;
    accesses: MemoryAccess[];
}

export interface MemoryAccess
{
    threadIdx: Dim3;
    blockIdx: Dim3;
    warpId: number;
    debugId: number;
    address: string;
    size: number;
    kind: AccessType;
    space: MemorySpace;
    typeIndex: number;
    timestamp: number;
}

export interface Dim3
{
    x: number;
    y: number;
    z: number;
}

export enum AccessType
{
    Read = 0,
    Write = 1
}

export enum MemorySpace
{
    Global = 0,
    Shared = 1,
    Constant = 2
}
