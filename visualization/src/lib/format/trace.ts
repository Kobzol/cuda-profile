import {MemoryAllocation} from './memory-allocation';
import {MemoryAccess} from './memory-access';

export interface Trace
{
    type: string;
    kernel: string;
    start: number;
    end: number;
    accesses: MemoryAccess[];
    allocations: MemoryAllocation[];
}
