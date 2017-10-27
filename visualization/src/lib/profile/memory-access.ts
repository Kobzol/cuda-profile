import {Dim3} from './dim3';
import {DebugLocation} from './metadata';

export enum AccessType
{
    Read = 0,
    Write = 1
}

export enum AddressSpace
{
    Global = 0,
    Shared = 1,
    Constant = 2
}

export interface MemoryAccessGroup
{
    key: string;
    size: number;
    kind: AccessType;
    space: AddressSpace;
    type: string;
    timestamp: number;
    location: DebugLocation | null;
    blockIdx: Dim3;
    warpId: number;
    accesses: MemoryAccess[];
}

export interface MemoryAccess
{
    id: number;
    threadIdx: Dim3;
    blockIdx: Dim3;
    warpId: number;
    address: string;
}
