import {AccessType, AddressSpace, Warp} from '../lib/profile/warp';

export function createWarp(params: Partial<Warp>): Warp
{
    return {
        key: params.key || '',
        index: params.index || 0,
        id: params.id || 0,
        slot: params.slot || 0,
        size: params.size || 4,
        accessType: params.accessType || AccessType.Read,
        space: params.space || AddressSpace.Global,
        type: params.type || '',
        timestamp: params.timestamp || '0',
        location: params.location || null,
        blockIdx: params.blockIdx || { x: 0, y: 0, z: 0 },
        accesses: params.accesses || []
    };
}
