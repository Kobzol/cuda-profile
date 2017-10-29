import bigInt, {BigInteger} from 'big-integer';
import {AddressRange} from '../trace/selection';
import {MemoryAccess} from './memory-access';
import {MemoryAllocation} from './memory-allocation';

function createRange(from: BigInteger, to: BigInteger): AddressRange
{
    return {
        from: '0x' + from.toString(16).toUpperCase(),
        to: '0x' + to.toString(16).toUpperCase()
    };
}

export function checkIntersection(rangeFrom: BigInteger, rangeTo: BigInteger,
                                  blockFrom: BigInteger, blockTo: BigInteger): boolean
{
    return !(blockFrom.geq(rangeTo) ||
            blockTo.leq(rangeFrom));
}

export function clampAddressRange(bound: AddressRange, range: AddressRange): AddressRange
{
    const from = bigInt.max(bigInt(bound.from.substr(2), 16), bigInt(range.from.substr(2), 16));
    const to = bigInt.min(bigInt(bound.to.substr(2), 16), bigInt(range.to.substr(2), 16));

    return createRange(from, to);
}

export function getAccessAddressRange(accesses: MemoryAccess[], size: number = 1): AddressRange
{
    let minAddress = bigInt('FFFFFFFFFFFFFFFF', 16);
    let maxAddress = bigInt('0', 16);

    for (const access of accesses)
    {
        const addressStart = bigInt(access.address.substr(2), 16);
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
    const from = bigInt(allocation.address.substr(2), 16);
    const to = from.add(allocation.size);

    return createRange(from, to);
}

export function getAddressRangeSize(range: AddressRange)
{
    return (bigInt(range.to.substr(2), 16).subtract(bigInt(range.from.substr(2), 16))).toJSNumber();
}
