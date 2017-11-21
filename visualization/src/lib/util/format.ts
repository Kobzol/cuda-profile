import {AccessType, AddressSpace} from '../profile/warp';
import {Dim3} from '../profile/dim3';

export function formatByteSize(value: number): string
{
    const sizes = [{
        label: 'GiB',
        exponent: 3
    }, {
        label: 'MiB',
        exponent: 2
    }, {
        label: 'KiB',
        exponent: 1
    }, {
        label: 'B',
        exponent: 0
    }];

    for (const size of sizes)
    {
        const power = Math.pow(1024, size.exponent);
        const scaled = value / power;

        if (scaled >= 1)
        {
            return `${Math.floor(scaled) === scaled ? scaled : scaled.toFixed(2)} ${size.label}`;
        }
    }

    return '0 B';
}
export function formatAddressSpace(space: AddressSpace): string
{
    switch (space)
    {
        case AddressSpace.Global: return 'global';
        case AddressSpace.Shared: return 'shared';
        case AddressSpace.Constant: return 'constant';
        default: return 'unknown';
    }
}
export function formatDim3(index: Dim3): string
{
    return `${index.z}.${index.y}.${index.x}`;
}
export function formatAccessType(access: AccessType): string
{
    return access === AccessType.Read ? 'read' : 'write';
}
