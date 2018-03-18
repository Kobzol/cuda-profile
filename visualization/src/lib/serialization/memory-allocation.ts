import {AddressSpace} from './memory-access';

export interface MemoryAllocation
{
    address: string;
    size: number;
    elementSize: number;
    space: AddressSpace;
    typeIndex: number;
    typeString: string;
    active: boolean;
    nameIndex: number;
    nameString: string;
    location: string;
}
