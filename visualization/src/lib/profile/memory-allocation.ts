import {AddressSpace} from './memory-access';

export interface MemoryAllocation
{
    address: string;
    size: number;
    elementSize: number;
    space: AddressSpace;
    type: string;
}
