import {MemoryAllocation} from './memory-allocation';
import {Dim3} from './dim3';
import {Warp} from './warp';

export interface Trace
{
    type: string;
    kernel: string;
    start: number;
    end: number;
    warps: Warp[];
    allocations: MemoryAllocation[];
    gridDim: Dim3;
    blockDim: Dim3;
    warpSize: number;
    bankSize: number;
}
