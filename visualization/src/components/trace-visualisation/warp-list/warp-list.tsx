import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/memory-access';
import {ThreadGrid} from '../thread-grid/thread-grid';

interface Props
{
    trace: Trace;
    warps: Warp[];
}

export class WarpList extends PureComponent<Props>
{
    render()
    {
        return (
            <div>
            {this.props.warps.map(warp =>
                <ThreadGrid
                    key={warp.key}
                    trace={this.props.trace}
                    warp={warp}
                    canvasDimensions={{width: 220, height: 40}}/>
                )}
            </div>
        );
    }
}
