import {
    checkIntersection, getIntersection, getAccessesAddressRange, getAddressRangeSize,
    getAllocationAddressRange, getWarpAccessesRange
} from '../lib/profile/address';
import bigInt from 'big-integer';
import {BigInteger} from 'big-integer';
import {MemoryAccess} from '../lib/profile/memory-access';
import {WarpAccess} from '../lib/trace/selection';
import {createWarp} from './util';

function num(value: string): BigInteger
{
    return bigInt(value, 16);
}

const from = num('FFFFFFFFFF100');
const to = num('FFFFFFFFFF200');

test('Address block should not intersect range if it\'s before', () => {
    expect(checkIntersection(from, to, num('FFFFFFFFFF090'), num('FFFFFFFFFF099'))).toBe(false);
    expect(checkIntersection(from, to, num('FFFFFFFFFF090'), num('FFFFFFFFFF100'))).toBe(false);
});
test('Address block should intersect range if it overlaps from left', () => {
    expect(checkIntersection(from, to, num('FFFFFFFFFF00'), num('FFFFFFFFFF101'))).toBe(true);
});
test('Address block should intersect range if it\'s inside', () => {
    expect(checkIntersection(from, to, num('FFFFFFFFFF100'), num('FFFFFFFFFF200'))).toBe(true);
});
test('Address block should intersect range if it overlaps from right', () => {
    expect(checkIntersection(from, to, num('FFFFFFFFFF150'), num('FFFFFFFFFF280'))).toBe(true);
});
test('Address block should intersect range if it covers the range', () => {
    expect(checkIntersection(from, to, num('FFFFFFFFFF000'), num('FFFFFFFFFF300'))).toBe(true);
});
test('Address block should not intersect range if it\'s after', () => {
    expect(checkIntersection(from, to, num('FFFFFFFFFF200'), num('FFFFFFFFFFB00'))).toBe(false);
    expect(checkIntersection(from, to, num('FFFFFFFFFF201'), num('FFFFFFFFFFB00'))).toBe(false);
});

test('Warp address range is calculated correctly', () => {
    const tid = {x: 0, y: 0, z: 0};
    const accesses: MemoryAccess[] = [{
        address: '0x100',
        id: 0,
        threadIdx: tid
    }, {
        address: '0xFF559A',
        id: 0,
        threadIdx: tid
    }, {
        address: '0x1555B79',
        id: 0,
        threadIdx: tid
    }, {
        address: '0x16',
        id: 0,
        threadIdx: tid
    }, {
        address: '0x1500',
        id: 0,
        threadIdx: tid
    }];

    expect(getAccessesAddressRange(accesses)).toEqual({
        from: '0x16',
        to: '0x1555B7A'
    });
});

const bound = {
    from: '0xFFFF100',
    to: '0xFFFF200'
};

test('Address should be clamped when overlapping from left', () => {
    const range = {
        from: '0xFFFF000',
        to: '0xFFFF150'
    };

    expect(getIntersection(bound, range)).toEqual({
        from: bound.from,
        to: range.to
    });
});
test('Address should be clamped when overlapping from right', () => {
    const range = {
        from: '0xFFFF150',
        to: '0xFFFF300'
    };

    expect(getIntersection(bound, range)).toEqual({
        from: range.from,
        to: bound.to
    });
});
test('Address should be clamped when inside', () => {
    const range = {
        from: '0xFFFF101',
        to: '0xFFFF199'
    };

    expect(getIntersection(bound, range)).toEqual({
        from: range.from,
        to: range.to
    });
});

test('Address size should be used for access address range calculation', () => {
    expect(getAccessesAddressRange([{
        address: '0xFFFF100',
        id: 0,
        threadIdx: {x: 0, y: 0, z: 0}
    }], 16)).toEqual({
        from: '0xFFFF100',
        to: '0xFFFF110'
    });
});
test('Address range size is calculated correctly ', () => {
    expect(getAddressRangeSize({
        from: '0xFFFF100',
        to: '0xFFFFAAA'
    })).toEqual(2474);
});
test('Allocation address range is calculated correctly ', () => {
    const address = '0xFFFFABC';

    expect(getAllocationAddressRange({
        address,
        size: 64,
        elementSize: 4,
        space: 0,
        type: '',
        name: '',
        location: ''
    })).toEqual({
        from: address,
        to: '0xFFFFAFC'
    });
});
test('Address range of warp accesses is calculated correctly', () => {
    const range = {
        from: '0x1000',
        to: '0x4000'
    };
    const warp = createWarp({
        size: 4
    });
    const createAccess = (address: string): WarpAccess =>
    {
        return {
            warp,
            access: {
                id: 0,
                threadIdx: {x: 0, y: 0, z: 0},
                address
            }
        };
    };

    const accesses = [
        createAccess('0x1050'),
        createAccess('0x5000'),
        createAccess('0x2000'),
        createAccess('0x4000'),
        createAccess('0x0800')
    ];

    expect(getWarpAccessesRange(range, accesses)).toEqual({
        from: '0x1050',
        to: '0x2004'
    });
});
