import {Warp} from './memory-access';
import {MemoryAllocation} from './memory-allocation';
import {Dim3} from './dim3';

export interface Trace
{
    start: number;
    end: number;
    warps: Warp[];
    allocations: MemoryAllocation[];
    gridDimension: Dim3;
    blockDimension: Dim3;
    warpSize: number;
}
