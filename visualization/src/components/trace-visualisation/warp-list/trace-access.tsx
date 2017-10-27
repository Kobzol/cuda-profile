import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {MemoryAccessGroup} from '../../../lib/profile/memory-access';
import {ThreadSelection} from '../thread-selection/thread-selection';

interface Props
{
    trace: Trace;
    warps: MemoryAccessGroup[];
}

export class WarpList extends PureComponent<Props>
{
    render()
    {
        return (
            <div>
            {this.props.warps.map(warp =>
                <ThreadSelection
                    key={warp.key}
                    bounds={this.props.trace.blockDimension}
                    accessGroup={warp} />
                )}
            </div>
        );
    }
}
