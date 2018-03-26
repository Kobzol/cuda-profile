import {Dim3} from './dim3';

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

export interface MemoryAccess
{
    threadIdx: Dim3;
    address: string;
    value: string;
}
