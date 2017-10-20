import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {MemoryAccessGroup} from '../../../lib/profile/memory-access';
import {ThreadSelection} from '../thread-selection/thread-selection';
import {MemoryMap} from '../memory-map/memory-map';

interface Props
{
    trace: Trace;
    accessGroup: MemoryAccessGroup;
}

export class TraceAccess extends PureComponent<Props>
{
    render()
    {
        /*return (
            <ThreadSelection
                bounds={this.props.trace.blockDimension}
                accessGroup={this.props.accessGroup} />
        );*/
        return (
            <MemoryMap
                allocations={this.props.trace.allocations}
                height={600} />
        );
    }
}
