import {getCtaId, getLaneId, getWarpId, getWarpStart} from '../lib/profile/api';
import {InvalidWarpData} from '../lib/profile/errors';

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
