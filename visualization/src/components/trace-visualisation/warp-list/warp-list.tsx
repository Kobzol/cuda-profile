import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {WarpGrid} from './warp-grid/warp-grid';
import {AddressRange, WarpAddressSelection} from '../../../lib/trace/selection';
import {getAccessAddressRange} from '../../../lib/profile/address';
import * as _ from 'lodash';

import './warp-list.css';

interface Props
{
    trace: Trace;
    warps: Warp[];
    selectRange: (range: WarpAddressSelection) => void;
    memorySelection: AddressRange;
}

export class WarpList extends PureComponent<Props>
{
    render()
    {
        return (
            <div className='warp-list'>
                {this.props.warps.length === 0 && 'No warps selected'}
                {this.props.warps.map(warp =>
                    <WarpGrid
                        key={warp.key}
                        trace={this.props.trace}
                        warp={warp}
                        canvasDimensions={{width: 260, height: 60}}
                        selectRange={this.handleRangeSelect}
                        memorySelection={this.props.memorySelection} />
                )}
            </div>
        );
    }

    handleRangeSelect = (range: WarpAddressSelection) =>
    {
        if (range !== null)
        {
            range.warpRange = getAccessAddressRange(_.flatMap(this.props.warps, warp => warp.accesses),
                this.props.warps[0].size);
        }
        this.props.selectRange(range);
    }
}
