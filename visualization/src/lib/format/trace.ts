import {MemoryAllocation} from './memory-allocation';
import {MemoryAccess} from './memory-access';
import {Dim3} from './dim3';

export interface Trace
{
    type: string;
    kernel: string;
    start: number;
    end: number;
    accesses: MemoryAccess[];
    allocations: MemoryAllocation[];
    gridDim: Dim3;
    blockDim: Dim3;
}
