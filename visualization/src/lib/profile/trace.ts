import {MemoryAccessGroup} from './memory-access';
import {MemoryAllocation} from './memory-allocation';
import {Dim3} from './dim3';

export interface Trace
{
    start: number;
    end: number;
    accessGroups: MemoryAccessGroup[];
    allocations: MemoryAllocation[];
    gridDimension: Dim3;
    blockDimension: Dim3;
}
