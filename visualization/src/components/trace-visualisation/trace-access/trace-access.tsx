import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {MemoryAccessGroup} from '../../../lib/profile/memory-access';
import {ThreadGrid} from '../thread-grid/thread-grid';

interface Props
{
    trace: Trace;
    accessGroup: MemoryAccessGroup;
}

export class TraceAccess extends PureComponent<Props>
{
    render()
    {
        return (
            <ThreadGrid
                id='thread-grid'
                accessGroup={this.props.accessGroup} />
        );
    }
}
