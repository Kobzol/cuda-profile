import {AddressRange} from '../../lib/trace/selection';

export interface WarpAddressSelection
{
    warpRange: AddressRange;
    threadRange: AddressRange;
}
