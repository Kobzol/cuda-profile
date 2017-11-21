import bigInt, {BigInteger} from 'big-integer';
import {AddressRange, WarpAddressSelection} from '../trace/selection';
import {MemoryAccess} from './memory-access';
import {MemoryAllocation} from './memory-allocation';

function createRange(from: BigInteger, to: BigInteger): AddressRange
{
    return {
        from: numToAddress(from),
        to: numToAddress(to)
    };
}

export function checkIntersection(rangeFrom: BigInteger, rangeTo: BigInteger,
                                  blockFrom: BigInteger, blockTo: BigInteger): boolean
{
    return !(blockFrom.geq(rangeTo) ||
            blockTo.leq(rangeFrom));
}
export function checkIntersectionRange(range: AddressRange, address: string, size: number): boolean
{
    const rangeFrom = addressToNum(range.from);
    const rangeTo = addressToNum(range.to);
    const blockFrom = addressToNum(address);
    const blockTo = blockFrom.add(size);

    return checkIntersection(rangeFrom, rangeTo, blockFrom, blockTo);
}

/**
 * Returns the intersection between bound and range or bound if there is no intersection.
 * @param {AddressRange} bound
 * @param {AddressRange} range
 * @returns {AddressRange}
 */
export function getIntersection(bound: AddressRange, range: AddressRange): AddressRange
{
    if (!checkIntersection(
            addressToNum(range.from), addressToNum(range.to),
            addressToNum(bound.from), addressToNum(bound.to)))
    {
        return bound;
    }

    const from = bigInt.max(addressToNum(bound.from), addressToNum(range.from));
    const to = bigInt.min(addressToNum(bound.to), addressToNum(range.to));

    return createRange(from, to);
}

export function getSelectionRange(bound: AddressRange, selections: WarpAddressSelection[]): AddressRange
{
    return getIntersection(bound, selections.length > 0 ?
        selections[0].warpRange : bound);
}

export function getAccessAddressRange(access: MemoryAccess, size: number = 1): AddressRange
{
    return {
        from: access.address,
        to: numToAddress(addressToNum(access.address).add(size))
    };
}
export function getAccessesAddressRange(accesses: MemoryAccess[], size: number = 1): AddressRange
{
    let minAddress = bigInt('FFFFFFFFFFFFFFFF', 16);
    let maxAddress = bigInt('0', 16);

    for (const access of accesses)
    {
        const addressStart = addressToNum(access.address);
        const addressEnd = addressStart.add(size);
        if (addressStart.lt(minAddress))
        {
            minAddress = addressStart;
        }
        if (addressEnd.gt(maxAddress))
        {
            maxAddress = addressEnd;
        }
    }

    return createRange(minAddress, maxAddress);
}
export function getAllocationAddressRange(allocation: MemoryAllocation): AddressRange
{
    const from = addressToNum(allocation.address);
    const to = from.add(allocation.size);

    return createRange(from, to);
}

export function getAddressRangeSize(range: AddressRange): number
{
    return (addressToNum(range.to).subtract(addressToNum(range.from))).toJSNumber();
}

export function addressToNum(address: string): BigInteger
{
    return bigInt(address.substr(2), 16);
}
export function numToAddress(num: BigInteger): string
{
    return `0x${num.toString(16).toUpperCase()}`;
}
export function addressAddStr(address: string, value: number): string
{
    return numToAddress(addressToNum(address).add(value));
}
