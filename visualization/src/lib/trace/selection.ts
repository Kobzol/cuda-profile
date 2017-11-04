export interface TraceSelection
{
    kernel: number;
    trace: number;
}

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
