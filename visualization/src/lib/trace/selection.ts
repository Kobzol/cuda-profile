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

export interface WarpAddressSelection
{
    warpRange: AddressRange;
    threadRange: AddressRange;
}
