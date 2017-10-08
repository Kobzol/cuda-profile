export enum AccessType
{
    Read = 0,
    Write = 1
}

export interface Dim3
{
    x: number;
    y: number;
    z: number;
}

export enum AddressSpace
{
    Global = 0,
    Shared = 1,
    Constant = 2
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
    space: AddressSpace;
    typeIndex: number;
    timestamp: number;
}
