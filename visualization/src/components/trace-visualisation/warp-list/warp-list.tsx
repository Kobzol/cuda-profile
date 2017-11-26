import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {WarpGrid} from './warp-grid/warp-grid';
import {AddressRange, WarpAddressSelection} from '../../../lib/trace/selection';
import {getAccessesAddressRange} from '../../../lib/profile/address';
import * as _ from 'lodash';

import './warp-list.scss';

interface Props
{
    trace: Trace;
    warps: Warp[];
    selectRange: (range: WarpAddressSelection) => void;
    memorySelection: AddressRange;
    deselect: (warp: Warp) => void;
    selectAllWarpAccesses: (warp: Warp) => void;
}

export class WarpList extends PureComponent<Props>
{
    render()
    {
        return (
            <div className='warp-list'>
                <h3>Warps</h3>
                <div>
                    {this.props.warps.length === 0 && 'No warps selected'}
                    {this.props.warps.map(warp =>
                        <WarpGrid
                            key={warp.key}
                            trace={this.props.trace}
                            warp={warp}
                            canvasDimensions={{width: 260, height: 60}}
                            selectRange={this.handleRangeSelect}
                            memorySelection={this.props.memorySelection}
                            selectionEnabled={false}
                            deselect={this.props.deselect}
                            selectAllWarpAccesses={this.props.selectAllWarpAccesses} />
                    )}
                </div>
            </div>
        );
    }

    handleRangeSelect = (range: WarpAddressSelection) =>
    {
        if (range !== null)
        {
            range.warpRange = getAccessesAddressRange(_.flatMap(this.props.warps, warp => warp.accesses),
                this.props.warps[0].size);
        }
        this.props.selectRange(range);
    }
}
