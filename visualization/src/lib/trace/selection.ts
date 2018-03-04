import {Warp} from '../profile/warp';
import {MemoryAccess} from '../profile/memory-access';

export interface TraceSelection
{
    kernel: number;
    trace: number;
}

/**
 * Represents a range of adresses. The 'to' attribute is exclusive.
 */
export interface AddressRange
{
    from: string;
    to: string;
}

export interface WarpAccess
{
    warp: Warp;
    access: MemoryAccess;
}

export function createWarpAccess(warp: Warp, access: MemoryAccess): WarpAccess
{
    return {
        warp,
        access
    };
}
