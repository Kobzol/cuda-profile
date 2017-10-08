import {MemoryAccessGroup} from './memory-access';
import {MemoryAllocation} from './memory-allocation';

export interface Trace
{
    start: number;
    end: number;
    accessGroups: MemoryAccessGroup[];
    allocations: MemoryAllocation[];
}
