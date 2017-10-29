import {Dim3} from './dim3';

export interface MemoryAccess
{
    id: number;
    threadIdx: Dim3;
    address: string;
}
