import {MemoryAccess} from './memory-access';
import {Dim3} from './dim3';

export interface Warp
{
    accesses: MemoryAccess[];
    blockIdx: Dim3;
    warpId: number;
    debugId: number;
    size: number;
    kind: number;
    space: number;
    typeIndex: number;
    timestamp: string;
}
