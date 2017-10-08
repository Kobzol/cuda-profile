import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {MemoryAccessGroup} from '../../../lib/profile/memory-access';
import {ThreadSelection} from '../thread-selection/thread-selection';

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
            <ThreadSelection
                id='thread-grid'
                accessGroup={this.props.accessGroup} />
        );
    }
}
