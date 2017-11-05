import {MemoryAccess} from '../../../../lib/profile/memory-access';
import {createSelector} from 'reselect';
import {Dictionary} from 'lodash';
import {Warp} from '../../../../lib/profile/warp';

export function createBlockSelector()
{
    return createSelector(
        (accessGroup: Warp) => accessGroup,
        (accessGroup: Warp) => createBlockData(accessGroup)
    );
}

export function createBlockData(accessGroup: Warp): Dictionary<MemoryAccess>
{
    const data: Dictionary<MemoryAccess> = {};

    for (const access of accessGroup.accesses)
    {
        data[`${access.threadIdx.z}.${access.threadIdx.y}.${access.threadIdx.x}`] = access;
    }

    return data;
}
