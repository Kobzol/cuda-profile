import {
    AccessType, AddressSpace, getBlockId, getConflicts, getCtaId, getLaneId, getWarpId, getWarpStart,
    Warp
} from '../lib/profile/warp';
import {InvalidWarpData} from '../lib/profile/errors';
import {DebugLocation} from '../lib/profile/metadata';
import {Dim3} from '../lib/profile/dim3';
import {MemoryAccess} from '../lib/profile/memory-access';

function createWarp(params: {key?: string,
                    index?: number,
                    id?: number,
                    slot?: number,
                    size?: number,
                    accessType?: AccessType,
                    space?: AddressSpace,
                    type?: string,
                    timestamp?: string,
                    location?: DebugLocation | null,
                    blockIdx?: Dim3,
                    accesses?: MemoryAccess[]}): Warp
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

test('Warp start is calculated correctly', () => {
    expect(getWarpStart(1, 32, {
        x: 6,
        y: 6,
        z: 4
    })).toEqual({
        x: 2,
        y: 5,
        z: 0
    });

    expect(getWarpStart(0, 32, {
        x: 6,
        y: 6,
        z: 4
    })).toEqual({
        x: 0,
        y: 0,
        z: 0
    });

    expect(getWarpStart(13, 32, {
        x: 13,
        y: 12,
        z: 11
    })).toEqual({
        x: 0,
        y: 8,
        z: 2
    });
});
test('CTA id is calculated correctly', () => {
    expect(getCtaId({
        x: 13,
        y: 14,
        z: 1
    }, {
        x: 155,
        y: 18,
        z: 10
    })).toEqual(4973);

    expect(getCtaId({
        x: 0,
        y: 0,
        z: 0
    }, {
        x: 11,
        y: 12,
        z: 13
    })).toEqual(0);
});
test('Lane id is calculated correctly', () => {
    expect(getLaneId({
        x: 2,
        y: 0,
        z: 1
    }, {
        x: 2,
        y: 5,
        z: 0
    }, {
        x: 6,
        y: 6,
        z: 4
    })).toEqual(6);

    const blockDim = {x: 256, y: 1, z: 1};
    const start = getWarpStart(44, 32, blockDim);
    expect(() => getLaneId({
        x: 160,
        y: 0,
        z: 0
    }, start, blockDim)).toThrow(InvalidWarpData);

});
test('Warp id is calculated correctly', () => {
    expect(getWarpId({
        x: 2,
        y: 0,
        z: 1
    }, 32, {
        x: 16,
        y: 18,
        z: 2
    })).toEqual(9);
});
test('Block id is calculated correctly', () => {
    expect(getBlockId({
        x: 2,
        y: 1,
        z: 1
    }, {
        x: 16,
        y: 18,
        z: 2
    })).toEqual(306);
});
test('Conflict in one warp is calculated correctly', () => {
    const warp: Warp = createWarp({
        size: 4,
        accesses: [
            {
                id: 0,
                threadIdx: {z: 0, y: 0, x: 0},
                address: '0xAA04'
            },
            {
                id: 1,
                threadIdx: {z: 0, y: 0, x: 1},
                address: '0xAA07'
            },
            {
                id: 2,
                threadIdx: {z: 0, y: 0, x: 2},
                address: '0xAA05'
            }
        ]
    });

    const conflicts = getConflicts([warp]);
    expect(conflicts.length).toEqual(4);
    expect(conflicts[0].address).toEqual('0xAA05');
    expect(conflicts[0].accesses.length).toEqual(2);
    expect(conflicts[0].accesses).toContainEqual({ warp, access: warp.accesses[0] });
    expect(conflicts[0].accesses).toContainEqual({ warp, access: warp.accesses[2] });

    expect(conflicts[1].address).toEqual('0xAA06');
    expect(conflicts[1].accesses.length).toEqual(2);
    expect(conflicts[1].accesses).toContainEqual({ warp, access: warp.accesses[0] });
    expect(conflicts[1].accesses).toContainEqual({ warp, access: warp.accesses[2] });

    expect(conflicts[2].address).toEqual('0xAA07');
    expect(conflicts[2].accesses.length).toEqual(3);
    expect(conflicts[2].accesses).toContainEqual({ warp, access: warp.accesses[0] });
    expect(conflicts[2].accesses).toContainEqual({ warp, access: warp.accesses[1] });
    expect(conflicts[2].accesses).toContainEqual({ warp, access: warp.accesses[2] });

    expect(conflicts[3].address).toEqual('0xAA08');
    expect(conflicts[3].accesses.length).toEqual(2);
    expect(conflicts[3].accesses).toContainEqual({ warp, access: warp.accesses[1] });
    expect(conflicts[3].accesses).toContainEqual({ warp, access: warp.accesses[2] });
});
