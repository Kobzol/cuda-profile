import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {WarpGrid} from './thread-grid/warp-grid';
import {WarpAddressSelection} from '../../../lib/trace/selection';

interface Props
{
    trace: Trace;
    warps: Warp[];
    selectRange: (range: WarpAddressSelection) => void;
}

export class WarpList extends PureComponent<Props>
{
    render()
    {
        return (
            <div>
                {this.props.warps.length === 0 && 'No warps selected'}
                {this.props.warps.map(warp =>
                    <WarpGrid
                        key={warp.key}
                        trace={this.props.trace}
                        warp={warp}
                        canvasDimensions={{width: 220, height: 40}}
                        selectRange={this.props.selectRange} />
                )}
            </div>
        );
    }
}
