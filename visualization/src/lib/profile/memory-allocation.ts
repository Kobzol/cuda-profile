import {AddressSpace} from './warp';

export interface MemoryAllocation
{
    address: string;
    size: number;
    elementSize: number;
    space: AddressSpace;
    type: string;
    name: string;
    location: string;
}
